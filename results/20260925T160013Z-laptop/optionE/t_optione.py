"""Option E: can TorchCodec 0.11 decode a frame of an MP4 without reading its moov?"""
import io, json, subprocess, sys, random
sys.path.insert(0, '/Users/HCornier/Documents/Personal/LeRobot/data11-bench')
import av, torch, torchcodec
from torchcodec.decoders import VideoDecoder
from readers import top_boxes, parse_moov
P = sys.argv[1]
data = open(P, 'rb').read()
size = len(data)
boxes = top_boxes(lambda o, n: data[o:o+n], size)
mo, ms = [(o, s) for t, o, s in boxes if t == 'moov'][0]
tr = list(parse_moov(data[mo:mo+ms]).values())[0]
print('torchcodec', torchcodec.__version__, 'file', size, 'boxes', [(t, o, s) for t, o, s in boxes], 'moov', mo, ms)

class Spy(io.RawIOBase):
    def __init__(self, b, blank=None):
        self.b, self.p, self.reads, self.blank = b, 0, [], blank
    def readable(self): return True
    def seekable(self): return True
    def read(self, n=-1):
        n = len(self.b) - self.p if n is None or n < 0 else n
        out = bytearray(self.b[self.p:self.p+n])
        if self.blank:
            a, e = self.blank
            for i in range(max(a, self.p), min(e, self.p + len(out))):
                out[i - self.p] = 0
        self.reads.append((self.p, len(out))); self.p += len(out); return bytes(out)
    def readinto(self, buf):
        d = self.read(len(buf)); buf[:len(d)] = d; return len(d)
    def seek(self, o, w=0):
        self.p = o if w == 0 else (self.p + o if w == 1 else len(self.b) + o); return self.p
    def tell(self): return self.p

def moov_read(reads):
    return sum(max(0, min(o + n, mo + ms) - max(o, mo)) for o, n in reads)

ref = VideoDecoder(P, seek_mode='approximate')
f = 301
want = ref.get_frames_at(indices=[f]).data
R = {}
# A. default open: how many moov bytes are read
s = Spy(data); d = VideoDecoder(s, seek_mode='approximate'); n_open = len(s.reads); x = d.get_frames_at(indices=[f]).data
R['A_default'] = {'moov_bytes_read': moov_read(s.reads), 'moov_size': ms, 'reads': len(s.reads), 'bit_exact': torch.equal(x, want)}
# B. custom_frame_mappings (ffprobe json): moov still read?
fm = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_entries', 'frame=pts,duration,key_frame', '-of', 'json', P], capture_output=True, text=True).stdout
try:
    s = Spy(data); d = VideoDecoder(s, custom_frame_mappings=fm); x = d.get_frames_at(indices=[f]).data
    R['B_custom_frame_mappings'] = {'moov_bytes_read': moov_read(s.reads), 'bit_exact': torch.equal(x, want)}
except Exception as e:
    R['B_custom_frame_mappings'] = {'error': f'{type(e).__name__}: {str(e)[:300]}'}
# C. moov bytes zeroed (as if never fetched), with and without custom_frame_mappings
for name, kw in (('C_moov_zeroed', dict(seek_mode='approximate')), ('C_moov_zeroed_custom_mappings', dict(custom_frame_mappings=fm))):
    try:
        s = Spy(data, blank=(mo + 8, mo + ms)); d = VideoDecoder(s, **kw); x = d.get_frames_at(indices=[f]).data
        R[name] = {'decoded': True, 'bit_exact': torch.equal(x, want)}
    except Exception as e:
        R[name] = {'decoded': False, 'error': f'{type(e).__name__}: {str(e)[:200]}'}
# D. no moov at all: only ftyp + mdat bytes of the needed packets (a truncated file): does the demuxer find frames?
kf = max(i for i in tr['sync'] if i <= f)
a, e = tr['offsets'][kf], tr['offsets'][f] + tr['sizes'][f]
ftyp = data[:[o + s_ for t, o, s_ in boxes if t == 'ftyp'][0]]
mdat_payload = data[a:e]
blob = ftyp + (8 + len(mdat_payload)).to_bytes(4, 'big') + b'mdat' + mdat_payload
try:
    d = VideoDecoder(Spy(blob), seek_mode='approximate'); R['D_ftyp_mdat_only'] = {'decoded': True, 'frames': d.metadata.num_frames}
except Exception as e2:
    R['D_ftyp_mdat_only'] = {'decoded': False, 'error': f'{type(e2).__name__}: {str(e2)[:200]}'}
# E. synthetic mini-MP4: packets keyframe..f cut from the raw bytes with an EXTERNAL index (offsets, sizes, sync:
#    ~8 bytes per frame, e.g. a parquet column) + codec extradata (av1C, stored once) -> re-muxed in memory -> TorchCodec
with av.open(P) as c:
    st = c.streams.video[0]
    extradata = bytes(st.codec_context.extradata or b'')
    buf = io.BytesIO()
    out = av.open(buf, 'w', format='mp4')
    os_ = out.add_stream('libsvtav1', rate=30)  # muxing only: no encoding happens, codec params set below
    os_.width, os_.height, os_.pix_fmt = st.codec_context.width, st.codec_context.height, 'yuv420p'
    os_.time_base = st.time_base
    os_.codec_context.extradata = extradata
    for j, i in enumerate(range(kf, f + 1)):
        pk = av.Packet(data[tr['offsets'][i]: tr['offsets'][i] + tr['sizes'][i]])
        pk.pts = pk.dts = j * 512; pk.time_base = st.time_base; pk.is_keyframe = (i in tr['sync']); pk.stream = os_
        out.mux(pk)
    out.close()
mini = buf.getvalue()
d = VideoDecoder(io.BytesIO(mini), seek_mode='approximate')
x = d.get_frames_at(indices=[f - kf]).data
R['E_synthetic_mini_mp4'] = {'packet_bytes': e - a, 'extradata_bytes': len(extradata), 'mini_file_bytes': len(mini), 'bit_exact': torch.equal(x, want),
                             'frames_in_mini': d.metadata.num_frames}
# F. raw AV1 OBU stream (no container at all): sequence header from av1C configOBUs + temporal units
cfg_obus = extradata[4:]
raw = cfg_obus + b''.join(data[tr['offsets'][i]: tr['offsets'][i] + tr['sizes'][i]] for i in range(kf, f + 1))
try:
    d = VideoDecoder(io.BytesIO(raw), seek_mode='approximate')
    fr = d.get_frames_at(indices=[f - kf]).data
    R['F_raw_obu_stream'] = {'decoded': True, 'bit_exact': torch.equal(fr, want), 'bytes': len(raw)}
except Exception as e3:
    R['F_raw_obu_stream'] = {'decoded': False, 'error': f'{type(e3).__name__}: {str(e3)[:200]}'}
print(json.dumps(R, indent=1))

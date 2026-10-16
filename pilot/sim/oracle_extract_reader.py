import io, time, urllib.request

class R(io.RawIOBase):
    def __init__(s, url):
        s.url, s.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r: s.size = int(r.headers["Content-Length"])
    def readable(s): return True
    def seekable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0): s.pos = {0: o, 1: s.pos + o, 2: s.size + o}[w]; return s.pos
    def readinto(s, b):
        n = min(len(b), s.size - s.pos)
        if n <= 0: return 0
        for a in range(5):
            try:
                req = urllib.request.Request(s.url, headers={"Range": f"bytes={s.pos}-{s.pos+n-1}"})
                with urllib.request.urlopen(req) as r: d = r.read()
                break
            except Exception:
                time.sleep(2 ** a)
        b[:len(d)] = d; s.pos += len(d); return len(d)


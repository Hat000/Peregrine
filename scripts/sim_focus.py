"""Robust sim-window focus + key sender + window screenshot.

Usage:
  sim_focus.py status              -- list candidate windows + foreground state
  sim_focus.py focus               -- restore + force-foreground the sim window
  sim_focus.py keys enter enter    -- focus, then send keys (enter/esc/down/up)
  sim_focus.py shot out.png        -- screenshot the sim window client area
"""
import ctypes
import ctypes.wintypes as wt
import sys
import time

user32 = ctypes.windll.user32
VK = {"enter": 0x0D, "esc": 0x1B, "down": 0x28, "up": 0x26, "left": 0x25, "right": 0x27}
KEYUP = 0x0002
SW_RESTORE = 9


def find_windows():
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _lp):
        n = user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            t = buf.value.lower()
            if "ai-gp" in t or "flightsim" in t or "ai grand prix" in t:
                found.append((hwnd, buf.value, bool(user32.IsWindowVisible(hwnd)),
                              bool(user32.IsIconic(hwnd))))
        return True

    user32.EnumWindows(_cb, None)
    return found


def force_foreground(hwnd) -> bool:
    user32.ShowWindow(hwnd, SW_RESTORE)
    time.sleep(0.3)
    # ALT keypress unlocks SetForegroundWindow from a background process
    user32.keybd_event(0x12, 0, 0, 0)
    user32.SetForegroundWindow(hwnd)
    user32.keybd_event(0x12, 0, KEYUP, 0)
    time.sleep(0.4)
    return user32.GetForegroundWindow() == hwnd


def send_key(vk):
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.06)
    user32.keybd_event(vk, 0, KEYUP, 0)
    time.sleep(0.35)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    wins = find_windows()
    if cmd == "status":
        fg = user32.GetForegroundWindow()
        for h, t, vis, icon in wins:
            print(f"hwnd={h} title='{t}' visible={vis} minimized={icon} "
                  f"foreground={h == fg}")
        if not wins:
            print("no sim window found")
        return 0
    if not wins:
        print("no sim window found")
        return 1
    hwnd = wins[0][0]
    if cmd == "focus":
        ok = force_foreground(hwnd)
        print(f"foreground={'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    if cmd == "keys":
        for attempt in range(3):
            if force_foreground(hwnd):
                break
            time.sleep(0.5)
        else:
            print("could not foreground the sim window")
            return 1
        for name in sys.argv[2:]:
            send_key(VK[name.lower()])
            print(f"sent {name}")
        return 0
    if cmd == "shot":
        out = sys.argv[2] if len(sys.argv) > 2 else "sim_window.png"
        r = wt.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        import PIL.ImageGrab as ig
        img = ig.grab(bbox=(r.left, r.top, r.right, r.bottom))
        img.save(out)
        print(f"saved {out}  rect=({r.left},{r.top},{r.right},{r.bottom})")
        return 0
    print(f"unknown cmd {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

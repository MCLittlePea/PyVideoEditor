import tkinter as tk
from tkinter import filedialog, ttk, messagebox, colorchooser
import cv2

# 强制单线程解码，彻底解决 FFmpeg 多线程断言崩溃问题
cv2.setNumThreads(1)
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, TextClip, CompositeAudioClip, \
    vfx, afx
import os
import sys
import threading
import queue
import time

# ========== 全局配置 ==========
PROXY_W = 960
PROXY_H = 540
EXPORT_W = 1920
EXPORT_H = 1080
MAX_CACHE_FRAMES = 5
TRACK_HEIGHT = 36
TIMELINE_PADDING = 80


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    elif hasattr(sys, '_NUITKA_ONEFILE_PARENT'):
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    return os.path.abspath(".")


ffmpeg_path = get_resource_path("imageio/ffmpeg/ffmpeg-win64-v4.2.2.exe")
if os.path.exists(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path


# ========== 基础片段类 ==========
class BaseClip:
    def __init__(self, name, duration):
        self.name = name
        self.start = 0.0  # 在时间轴上的起始时间
        self.clip_in = 0.0  # 素材内部入点
        self.clip_out = duration  # 素材内部出点
        self.duration = duration  # 素材总时长
        self.track = 0
        self.selected = False
        self.opacity = 1.0
        self.x = 0
        self.y = 0
        self.scale = 1.0
        self.rotation = 0

    @property
    def length(self):
        return self.clip_out - self.clip_in

    def get_frame(self, local_time):
        return None


class VideoClipItem(BaseClip):
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / self.fps
        super().__init__(os.path.basename(path), duration)
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.speed = 1.0
        self.reverse = False
        self.volume = 1.0
        self.filter = "无"
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self._cache_idx = -1
        self._cache_frame = None
        self.scale = 0.5
        self.x = 0
        self.y = 0

    def get_frame(self, local_time):
        if local_time < 0 or local_time > self.length:
            return None
        src_time = self.clip_in + local_time * self.speed
        if self.reverse:
            src_time = self.duration - src_time
        frame_idx = int(src_time * self.fps)
        if frame_idx == self._cache_idx and self._cache_frame is not None:
            return self._cache_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(int(self.duration * self.fps) - 1, frame_idx)))
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        w = int(self.orig_w * self.scale * (PROXY_W / EXPORT_W))
        h = int(self.orig_h * self.scale * (PROXY_H / EXPORT_H))
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        # 颜色调节
        if self.brightness != 1.0 or self.contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=(self.brightness - 1) * 128)
        if self.saturation != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= self.saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 滤镜
        if self.filter == "黑白":
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        elif self.filter == "复古":
            frame = frame.astype(np.float32)
            frame[:, :, 0] *= 0.9
            frame[:, :, 1] *= 0.8
            frame[:, :, 2] *= 0.65
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        elif self.filter == "冷色调":
            frame = frame.astype(np.float32)
            frame[:, :, 0] *= 1.15
            frame[:, :, 2] *= 0.85
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        elif self.filter == "暖色调":
            frame = frame.astype(np.float32)
            frame[:, :, 0] *= 0.85
            frame[:, :, 2] *= 1.2
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if self.opacity < 1.0:
            frame = (frame * self.opacity).astype(np.uint8)
        self._cache_idx = frame_idx
        self._cache_frame = frame
        return frame

    def get_moviepy_clip(self):
        clip = VideoFileClip(self.path).subclip(self.clip_in, self.clip_out)
        if self.reverse:
            clip = clip.fx(vfx.time_mirror)
        if self.speed != 1.0:
            clip = clip.fx(vfx.speedx, self.speed)
        clip = clip.resize((int(self.orig_w * self.scale), int(self.orig_h * self.scale)))
        clip = clip.set_position((self.x, self.y))
        clip = clip.set_opacity(self.opacity)
        clip = clip.set_start(self.start)
        if self.volume != 1.0:
            clip = clip.fx(afx.volumex, self.volume)

        # 颜色与滤镜
        def apply_color(frame):
            if self.brightness != 1.0 or self.contrast != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=(self.brightness - 1) * 128)
            if self.saturation != 1.0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= self.saturation
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            if self.filter == "黑白":
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            elif self.filter == "复古":
                f = frame.astype(np.float32)
                f[:, :, 0] *= 0.9;
                f[:, :, 1] *= 0.8;
                f[:, :, 2] *= 0.65
                frame = np.clip(f, 0, 255).astype(np.uint8)
            elif self.filter == "冷色调":
                f = frame.astype(np.float32)
                f[:, :, 0] *= 1.15;
                f[:, :, 2] *= 0.85
                frame = np.clip(f, 0, 255).astype(np.uint8)
            elif self.filter == "暖色调":
                f = frame.astype(np.float32)
                f[:, :, 0] *= 0.85;
                f[:, :, 2] *= 1.2
                frame = np.clip(f, 0, 255).astype(np.uint8)
            return frame

        clip = clip.fl_image(apply_color)
        return clip

    def release(self):
        if self.cap:
            self.cap.release()


class ImageClipItem(BaseClip):
    def __init__(self, path, duration=5.0):
        self.path = path
        img = Image.open(path).convert("RGB")
        self.orig_img = np.array(img)
        super().__init__(os.path.basename(path), duration)
        self.scale = 0.3
        self.x = 100
        self.y = 100

    def get_frame(self, local_time):
        if local_time < 0 or local_time > self.length:
            return None
        h, w = self.orig_img.shape[:2]
        new_w = int(w * self.scale * (PROXY_W / EXPORT_W))
        new_h = int(h * self.scale * (PROXY_H / EXPORT_H))
        img = cv2.resize(self.orig_img, (new_w, new_h))
        if self.opacity < 1.0:
            img = (img * self.opacity).astype(np.uint8)
        return img

    def get_moviepy_clip(self):
        clip = ImageClip(self.path).set_duration(self.length)
        clip = clip.resize((int(self.orig_img.shape[1] * self.scale), int(self.orig_img.shape[0] * self.scale)))
        clip = clip.set_position((self.x, self.y)).set_opacity(self.opacity).set_start(self.start)
        return clip


class TextClipItem(BaseClip):
    def __init__(self, text="新建文本", duration=5.0):
        super().__init__("文本", duration)
        self.text = text
        self.font_size = 48
        self.color = "#FFFFFF"
        self.x = PROXY_W // 2
        self.y = 100
        self._cache = None

    def get_frame(self, local_time):
        if local_time < 0 or local_time > self.length:
            return None
        if self._cache is not None:
            return self._cache
        img = Image.new("RGBA", (len(self.text) * self.font_size, self.font_size + 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", self.font_size)
        except:
            font = ImageFont.load_default()
        draw.text((0, 0), self.text, fill=self.color, font=font)
        self._cache = np.array(img.convert("RGB"))
        return self._cache

    def get_moviepy_clip(self):
        clip = TextClip(self.text, fontsize=self.font_size, color=self.color, font="simhei")
        clip = clip.set_duration(self.length).set_position((self.x, self.y)).set_start(self.start)
        return clip


class AudioClipItem(BaseClip):
    def __init__(self, path):
        self.path = path
        clip = AudioFileClip(path)
        super().__init__(os.path.basename(path), clip.duration)
        clip.close()
        self.volume = 1.0
        self.fade_in = 0.0
        self.fade_out = 0.0

    def get_moviepy_clip(self):
        clip = AudioFileClip(self.path).subclip(self.clip_in, self.clip_out)
        clip = clip.volumex(self.volume).set_start(self.start)
        if self.fade_in > 0:
            clip = clip.audio_fadein(self.fade_in)
        if self.fade_out > 0:
            clip = clip.audio_fadeout(self.fade_out)
        return clip


# ========== 播放引擎（墙钟时间同步，精准无漂移） ==========
class PlayerEngine(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.clips = []
        self.audio_clips = []
        self.playing = False
        self.stop_flag = threading.Event()
        self.frame_queue = queue.Queue(maxsize=3)
        self.current_time = 0.0
        self.total_duration = 10.0
        self._play_start_time = 0.0
        self._pause_time = 0.0

    def update_clips(self, clips, audio_clips):
        self.clips = clips
        self.audio_clips = audio_clips
        all_end = [c.start + c.length for c in clips] + [a.start + a.length for a in audio_clips] + [10.0]
        self.total_duration = max(all_end)

    def seek(self, time):
        self.current_time = max(0.0, min(time, self.total_duration))
        self._pause_time = self.current_time
        if self.playing:
            self._play_start_time = time.time() - self.current_time
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._play_start_time = time.time() - self._pause_time
        else:
            self._pause_time = self.current_time
        return self.playing

    def run(self):
        while not self.stop_flag.is_set():
            if not self.playing:
                time.sleep(0.02)
                continue

            # 以系统真实时间为准，保证播放速率1:1
            elapsed = time.time() - self._play_start_time
            self.current_time = max(0.0, min(elapsed, self.total_duration))

            if self.current_time >= self.total_duration:
                self.playing = False
                self._pause_time = 0.0
                continue

            # 合成画面
            canvas = np.zeros((PROXY_H, PROXY_W, 3), dtype=np.uint8)
            sorted_clips = sorted(self.clips, key=lambda c: c.track)
            for clip in sorted_clips:
                local_t = self.current_time - clip.start
                frame = clip.get_frame(local_t)
                if frame is None:
                    continue
                fh, fw = frame.shape[:2]
                dx1, dy1 = max(0, clip.x), max(0, clip.y)
                dx2, dy2 = min(PROXY_W, dx1 + fw), min(PROXY_H, dy1 + fh)
                sx1, sy1 = dx1 - clip.x, dy1 - clip.y
                sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
                if dx2 > dx1 and dy2 > dy1:
                    canvas[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]

            try:
                self.frame_queue.put_nowait((self.current_time, canvas))
            except queue.Full:
                pass

            time.sleep(0.008)  # 约120次/秒轮询，保证时间精度

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=1)


# ========== 主应用 ==========
class VideoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("专业视频编辑器 - 多轨道非线性剪辑")
        self.root.geometry("1500x950")
        self.root.configure(bg="#1a1a1a")

        self.video_clips = []
        self.audio_clips = []
        self.selected_clip = None
        self.selected_audio_clip = None
        self.timeline_zoom = 40
        self.playhead_x = 0
        self.drag_mode = None
        self.drag_clip = None
        self.drag_offset = 0

        self.player = PlayerEngine()
        self.player.start()

        self._build_ui()
        self._preview_loop()
        self._timeline_update_loop()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # 顶部工具栏
        toolbar = tk.Frame(self.root, bg="#2b2b2b", height=44)
        toolbar.pack(fill=tk.X)
        btns = [
            ("导入视频", self._import_video),
            ("导入图片/贴纸", self._import_image),
            ("导入音频", self._import_audio),
            ("添加文本", self._add_text),
            ("分割片段", self._split_clip),
            ("删除选中", self._delete_clip),
        ]
        for text, cmd in btns:
            tk.Button(toolbar, text=text, command=cmd, bg="#3a3a3a", fg="white",
                      relief=tk.FLAT, padx=12, pady=6).pack(side=tk.LEFT, padx=3, pady=6)
        tk.Button(toolbar, text="导出视频", command=self._export, bg="#0078d4", fg="white",
                  relief=tk.FLAT, padx=16, pady=6).pack(side=tk.RIGHT, padx=10, pady=6)

        # 主分割
        main_pw = tk.PanedWindow(self.root, orient=tk.VERTICAL, bg="#1a1a1a")
        main_pw.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 预览区 + 属性面板
        top_frame = tk.Frame(main_pw, bg="#000000")
        main_pw.add(top_frame, height=580)

        self.prop_frame = tk.Frame(top_frame, bg="#2b2b2b", width=240)
        self.prop_frame.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(self.prop_frame, text="属性面板", bg="#2b2b2b", fg="white",
                 font=("微软雅黑", 10, "bold")).pack(pady=8)
        self.prop_content = tk.Frame(self.prop_frame, bg="#2b2b2b")
        self.prop_content.pack(fill=tk.X, padx=10)
        self._update_prop_panel()

        self.preview_canvas = tk.Canvas(top_frame, bg="#000000", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.preview_canvas.bind("<ButtonPress-1>", self._preview_press)
        self.preview_canvas.bind("<B1-Motion>", self._preview_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._preview_release)

        # 播放控制栏
        ctrl_bar = tk.Frame(top_frame, bg="#2b2b2b", height=36)
        ctrl_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.play_btn = tk.Button(ctrl_bar, text="▶ 播放", command=self._toggle_play,
                                  bg="#3a3a3a", fg="white", relief=tk.FLAT, width=10)
        self.play_btn.pack(side=tk.LEFT, padx=10, pady=5)
        self.time_label = tk.Label(ctrl_bar, text="00:00.0 / 00:00.0", bg="#2b2b2b", fg="white")
        self.time_label.pack(side=tk.LEFT, padx=10)

        # 时间轴区域
        timeline_frame = tk.Frame(main_pw, bg="#2b2b2b", height=280)
        main_pw.add(timeline_frame)

        self.timeline_canvas = tk.Canvas(timeline_frame, bg="#1e1e1e", highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.BOTH, expand=True)
        self.timeline_canvas.bind("<ButtonPress-1>", self._timeline_press)
        self.timeline_canvas.bind("<B1-Motion>", self._timeline_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self._timeline_release)
        self.timeline_canvas.bind("<MouseWheel>", self._timeline_zoom)

    # ========== 导入功能 ==========
    def _import_video(self):
        path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv")])
        if not path:
            return
        clip = VideoClipItem(path)
        clip.track = len(self.video_clips)
        clip.y = 0 if clip.track == 0 else 50
        self.video_clips.append(clip)
        self._refresh_player()
        self._redraw_timeline()

    def _import_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not path:
            return
        clip = ImageClipItem(path)
        clip.track = len(self.video_clips)
        self.video_clips.append(clip)
        self._refresh_player()
        self._redraw_timeline()

    def _import_audio(self):
        path = filedialog.askopenfilename(filetypes=[("音频文件", "*.mp3 *.wav *.m4a *.flac")])
        if not path:
            return
        audio = AudioClipItem(path)
        self.audio_clips.append(audio)
        self._refresh_player()
        self._redraw_timeline()

    def _add_text(self):
        clip = TextClipItem()
        clip.track = len(self.video_clips)
        self.video_clips.append(clip)
        self._refresh_player()
        self._redraw_timeline()

    # ========== 编辑功能 ==========
    def _split_clip(self):
        if not self.selected_clip:
            messagebox.showinfo("提示", "请先在时间轴选中一个视频/图片/文本片段")
            return
        c = self.selected_clip
        split_local = self.player.current_time - c.start
        if split_local <= 0 or split_local >= c.length:
            return
        new_clip_out = c.clip_in + split_local
        old_clip_in = new_clip_out

        new_clip = type(c)(c.path if hasattr(c, 'path') else c.text)
        new_clip.__dict__.update(c.__dict__)
        new_clip.clip_in = old_clip_in
        new_clip.clip_out = c.clip_out
        new_clip.start = self.player.current_time
        new_clip.selected = False

        c.clip_out = new_clip_out
        self.video_clips.append(new_clip)
        self._refresh_player()
        self._redraw_timeline()

    def _delete_clip(self):
        if self.selected_clip:
            if hasattr(self.selected_clip, 'release'):
                self.selected_clip.release()
            self.video_clips.remove(self.selected_clip)
            self.selected_clip = None
        elif self.selected_audio_clip:
            self.audio_clips.remove(self.selected_audio_clip)
            self.selected_audio_clip = None
        self._refresh_player()
        self._redraw_timeline()
        self._update_prop_panel()

    # ========== 播放控制 ==========
    def _toggle_play(self):
        playing = self.player.toggle_play()
        self.play_btn.config(text="⏸ 暂停" if playing else "▶ 播放")

    def _preview_loop(self):
        try:
            t, frame = self.player.frame_queue.get_nowait()
            self.playhead_x = t * self.timeline_zoom
            img = Image.fromarray(frame)
            self._preview_imgtk = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._preview_imgtk)

            if self.selected_clip and hasattr(self.selected_clip, 'get_frame'):
                c = self.selected_clip
                x, y = c.x, c.y
                f = c.get_frame(0)
                if f is not None:
                    w, h = f.shape[1], f.shape[0]
                    self.preview_canvas.create_rectangle(x, y, x + w, y + h, outline="#00ff00", width=2)
        except queue.Empty:
            pass
        self.root.after(16, self._preview_loop)

    def _timeline_update_loop(self):
        if self.player.playing:
            self._redraw_timeline()
            m, s = divmod(self.player.current_time, 60)
            mt, st = divmod(self.player.total_duration, 60)
            self.time_label.config(text=f"{int(m):02d}:{s:04.1f} / {int(mt):02d}:{st:04.1f}")
        self.root.after(100, self._timeline_update_loop)

    # ========== 预览区交互 ==========
    def _preview_press(self, event):
        for clip in reversed(self.video_clips):
            f = clip.get_frame(0)
            if f is None:
                continue
            w, h = f.shape[1], f.shape[0]
            if clip.x <= event.x <= clip.x + w and clip.y <= event.y <= clip.y + h:
                self.selected_clip = clip
                self.selected_audio_clip = None
                self.drag_mode = "move"
                self.drag_offset = (event.x - clip.x, event.y - clip.y)
                self._update_prop_panel()
                self._redraw_timeline()
                return
        self.selected_clip = None
        self._update_prop_panel()

    def _preview_drag(self, event):
        if self.drag_mode != "move" or not self.selected_clip:
            return
        self.selected_clip.x = max(0, event.x - self.drag_offset[0])
        self.selected_clip.y = max(0, event.y - self.drag_offset[1])

    def _preview_release(self, event):
        self.drag_mode = None

    # ========== 时间轴交互 ==========
    def _redraw_timeline(self):
        self.timeline_canvas.delete("all")
        total_w = int(self.player.total_duration * self.timeline_zoom) + 300
        video_track_count = max(len(self.video_clips), 1)
        audio_track_count = max(len(self.audio_clips), 1)
        total_tracks = video_track_count + audio_track_count + 1
        total_h = 30 + total_tracks * TRACK_HEIGHT + 20
        self.timeline_canvas.config(scrollregion=(0, 0, total_w, total_h))

        # 时间刻度
        step = 1 if self.timeline_zoom > 30 else 5
        for i in range(0, int(self.player.total_duration) + step, step):
            x = i * self.timeline_zoom
            self.timeline_canvas.create_line(x, 0, x, 30, fill="#555")
            self.timeline_canvas.create_text(x + 3, 12, text=f"{i}s", fill="#999", anchor=tk.W, font=("微软雅黑", 8))

        # 轨道线
        for i in range(total_tracks):
            y = 30 + i * TRACK_HEIGHT
            self.timeline_canvas.create_line(0, y, total_w, y, fill="#333")

        # 视频片段
        for idx, clip in enumerate(self.video_clips):
            y = 30 + idx * TRACK_HEIGHT + 3
            x = clip.start * self.timeline_zoom
            w = clip.length * self.timeline_zoom
            color = "#4a9eff" if clip != self.selected_clip else "#00ff88"
            self.timeline_canvas.create_rectangle(x, y, x + w, y + TRACK_HEIGHT - 6, fill=color, outline="")
            self.timeline_canvas.create_text(x + 6, y + (TRACK_HEIGHT - 6) // 2, text=clip.name, fill="white",
                                             anchor=tk.W, font=("微软雅黑", 8))
            # 左右拖拽手柄 - 修复为标准6位颜色
            self.timeline_canvas.create_rectangle(x, y, x + 6, y + TRACK_HEIGHT - 6, fill="#aaaaaa", outline="")
            self.timeline_canvas.create_rectangle(x + w - 6, y, x + w, y + TRACK_HEIGHT - 6, fill="#aaaaaa", outline="")

        # 音频片段
        audio_base = 30 + video_track_count * TRACK_HEIGHT
        for idx, clip in enumerate(self.audio_clips):
            y = audio_base + idx * TRACK_HEIGHT + 3
            x = clip.start * self.timeline_zoom
            w = clip.length * self.timeline_zoom
            color = "#ff6b6b" if clip != self.selected_audio_clip else "#ffaa00"
            self.timeline_canvas.create_rectangle(x, y, x + w, y + TRACK_HEIGHT - 6, fill=color, outline="")
            self.timeline_canvas.create_text(x + 6, y + (TRACK_HEIGHT - 6) // 2, text=clip.name, fill="white",
                                             anchor=tk.W, font=("微软雅黑", 8))

        # 播放头
        px = self.player.current_time * self.timeline_zoom
        self.timeline_canvas.create_line(px, 0, px, total_h, fill="#ff3333", width=2)
        self.timeline_canvas.create_polygon(px - 6, 0, px + 6, 0, px, 10, fill="#ff3333", outline="")

    def _timeline_press(self, event):
        x = self.timeline_canvas.canvasx(event.x)
        y = self.timeline_canvas.canvasy(event.y)

        # 检测点击片段
        video_track_count = len(self.video_clips)
        for idx, clip in enumerate(self.video_clips):
            cy = 30 + idx * TRACK_HEIGHT + 3
            cx = clip.start * self.timeline_zoom
            cw = clip.length * self.timeline_zoom
            if cy <= y <= cy + TRACK_HEIGHT - 6:
                if cx <= x <= cx + 6:
                    self.drag_mode = "resize_left"
                    self.drag_clip = clip
                    self.selected_clip = clip
                    self.selected_audio_clip = None
                    self._update_prop_panel()
                    return
                elif cx + cw - 6 <= x <= cx + cw:
                    self.drag_mode = "resize_right"
                    self.drag_clip = clip
                    self.selected_clip = clip
                    self.selected_audio_clip = None
                    self._update_prop_panel()
                    return
                elif cx <= x <= cx + cw:
                    self.drag_mode = "move_clip"
                    self.drag_clip = clip
                    self.drag_offset = x - cx
                    self.selected_clip = clip
                    self.selected_audio_clip = None
                    self._update_prop_panel()
                    return

        # 音频片段点击
        audio_base = 30 + video_track_count * TRACK_HEIGHT
        for idx, clip in enumerate(self.audio_clips):
            cy = audio_base + idx * TRACK_HEIGHT + 3
            cx = clip.start * self.timeline_zoom
            cw = clip.length * self.timeline_zoom
            if cy <= y <= cy + TRACK_HEIGHT - 6 and cx <= x <= cx + cw:
                self.drag_mode = "move_audio"
                self.drag_clip = clip
                self.drag_offset = x - cx
                self.selected_audio_clip = clip
                self.selected_clip = None
                self._update_prop_panel()
                return

        # 空白处：跳转播放头
        self.player.seek(x / self.timeline_zoom)
        self.selected_clip = None
        self.selected_audio_clip = None
        self._update_prop_panel()
        self._redraw_timeline()

    def _timeline_drag(self, event):
        x = self.timeline_canvas.canvasx(event.x)
        if self.drag_mode == "move_clip" and self.drag_clip:
            new_start = max(0, (x - self.drag_offset) / self.timeline_zoom)
            self.drag_clip.start = new_start
            self._refresh_player()
            self._redraw_timeline()
        elif self.drag_mode == "resize_left" and self.drag_clip:
            new_in = max(0, x / self.timeline_zoom - self.drag_clip.start)
            if new_in < self.drag_clip.length - 0.1:
                self.drag_clip.clip_in = self.drag_clip.clip_in + new_in
                self.drag_clip.start = x / self.timeline_zoom
                self._refresh_player()
                self._redraw_timeline()
        elif self.drag_mode == "resize_right" and self.drag_clip:
            new_end = (x / self.timeline_zoom) - self.drag_clip.start
            if new_end > 0.1:
                self.drag_clip.clip_out = self.drag_clip.clip_in + new_end
                self._refresh_player()
                self._redraw_timeline()
        elif self.drag_mode == "move_audio" and self.drag_clip:
            self.drag_clip.start = max(0, (x - self.drag_offset) / self.timeline_zoom)
            self._refresh_player()
            self._redraw_timeline()
        elif self.drag_mode is None:
            self.player.seek(x / self.timeline_zoom)
            self._redraw_timeline()

    def _timeline_release(self, event):
        self.drag_mode = None
        self.drag_clip = None

    def _timeline_zoom(self, event):
        if event.delta > 0:
            self.timeline_zoom = min(150, self.timeline_zoom + 8)
        else:
            self.timeline_zoom = max(10, self.timeline_zoom - 8)
        self._redraw_timeline()

    # ========== 属性面板 ==========
    def _update_prop_panel(self):
        for w in self.prop_content.winfo_children():
            w.destroy()

        clip = self.selected_clip if self.selected_clip else self.selected_audio_clip
        if not clip:
            tk.Label(self.prop_content, text="选中片段后编辑属性", bg="#2b2b2b", fg="#888").pack(pady=30)
            return

        # 通用属性
        tk.Label(self.prop_content, text="起始时间(s)", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
        start_var = tk.DoubleVar(value=round(clip.start, 2))
        tk.Entry(self.prop_content, textvariable=start_var).pack(fill=tk.X)

        tk.Label(self.prop_content, text="时长(s)", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
        len_var = tk.DoubleVar(value=round(clip.length, 2))
        tk.Entry(self.prop_content, textvariable=len_var).pack(fill=tk.X)

        # 视频/图片属性
        if hasattr(clip, 'x'):
            tk.Label(self.prop_content, text="位置 X", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(8, 0))
            x_var = tk.IntVar(value=clip.x)
            tk.Entry(self.prop_content, textvariable=x_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="位置 Y", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            y_var = tk.IntVar(value=clip.y)
            tk.Entry(self.prop_content, textvariable=y_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="缩放比例", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            scale_var = tk.DoubleVar(value=clip.scale)
            tk.Entry(self.prop_content, textvariable=scale_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="不透明度", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            op_var = tk.DoubleVar(value=clip.opacity)
            ttk.Scale(self.prop_content, from_=0.1, to=1.0, variable=op_var).pack(fill=tk.X)

        # 视频专属
        if isinstance(clip, VideoClipItem):
            tk.Label(self.prop_content, text="播放速度", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(8, 0))
            speed_var = tk.DoubleVar(value=clip.speed)
            tk.Entry(self.prop_content, textvariable=speed_var).pack(fill=tk.X)

            rev_var = tk.BooleanVar(value=clip.reverse)
            tk.Checkbutton(self.prop_content, text="倒放", variable=rev_var, bg="#2b2b2b", fg="white").pack(anchor=tk.W,
                                                                                                            pady=4)

            tk.Label(self.prop_content, text="滤镜", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            filter_var = tk.StringVar(value=clip.filter)
            filter_combo = ttk.Combobox(self.prop_content, textvariable=filter_var,
                                        values=["无", "黑白", "复古", "冷色调", "暖色调"])
            filter_combo.pack(fill=tk.X)

            tk.Label(self.prop_content, text="亮度", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            bri_var = tk.DoubleVar(value=clip.brightness)
            ttk.Scale(self.prop_content, from_=0.3, to=2.0, variable=bri_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="对比度", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            con_var = tk.DoubleVar(value=clip.contrast)
            ttk.Scale(self.prop_content, from_=0.3, to=2.0, variable=con_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="饱和度", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            sat_var = tk.DoubleVar(value=clip.saturation)
            ttk.Scale(self.prop_content, from_=0.0, to=2.0, variable=sat_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="音量", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            vol_var = tk.DoubleVar(value=clip.volume)
            ttk.Scale(self.prop_content, from_=0.0, to=2.0, variable=vol_var).pack(fill=tk.X)

        # 文本专属
        if isinstance(clip, TextClipItem):
            tk.Label(self.prop_content, text="文本内容", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(8, 0))
            text_var = tk.StringVar(value=clip.text)
            tk.Entry(self.prop_content, textvariable=text_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="字号", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            size_var = tk.IntVar(value=clip.font_size)
            tk.Entry(self.prop_content, textvariable=size_var).pack(fill=tk.X)

            def pick_color():
                c = colorchooser.askcolor(color=clip.color)[1]
                if c:
                    clip.color = c
                    clip._cache = None

            tk.Button(self.prop_content, text="选择颜色", command=pick_color).pack(fill=tk.X, pady=6)

        # 音频专属
        if isinstance(clip, AudioClipItem):
            tk.Label(self.prop_content, text="音量", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(8, 0))
            vol_var = tk.DoubleVar(value=clip.volume)
            ttk.Scale(self.prop_content, from_=0.0, to=2.0, variable=vol_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="淡入(s)", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            fi_var = tk.DoubleVar(value=clip.fade_in)
            tk.Entry(self.prop_content, textvariable=fi_var).pack(fill=tk.X)

            tk.Label(self.prop_content, text="淡出(s)", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(4, 0))
            fo_var = tk.DoubleVar(value=clip.fade_out)
            tk.Entry(self.prop_content, textvariable=fo_var).pack(fill=tk.X)

        def apply_all():
            clip.start = start_var.get()
            if hasattr(clip, 'x'):
                clip.x = x_var.get()
                clip.y = y_var.get()
                clip.scale = scale_var.get()
                clip.opacity = op_var.get()
            if isinstance(clip, VideoClipItem):
                clip.speed = speed_var.get()
                clip.reverse = rev_var.get()
                clip.filter = filter_var.get()
                clip.brightness = bri_var.get()
                clip.contrast = con_var.get()
                clip.saturation = sat_var.get()
                clip.volume = vol_var.get()
                clip._cache_frame = None
            if isinstance(clip, TextClipItem):
                clip.text = text_var.get()
                clip.font_size = size_var.get()
                clip._cache = None
            if isinstance(clip, AudioClipItem):
                clip.volume = vol_var.get()
                clip.fade_in = fi_var.get()
                clip.fade_out = fo_var.get()
            self._refresh_player()
            self._redraw_timeline()

        tk.Button(self.prop_content, text="应用修改", command=apply_all,
                  bg="#0078d4", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=12)

    # ========== 导出 ==========
    def _export(self):
        if not self.video_clips:
            messagebox.showwarning("提示", "请至少添加一个视频/图片片段")
            return
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4视频", "*.mp4")])
        if not path:
            return
        threading.Thread(target=self._export_thread, args=(path,), daemon=True).start()
        messagebox.showinfo("提示", "开始后台导出，完成后会弹窗提示")

    def _export_thread(self, path):
        try:
            video_clips = [c.get_moviepy_clip() for c in self.video_clips]
            final_video = CompositeVideoClip(video_clips, size=(EXPORT_W, EXPORT_H))

            if self.audio_clips:
                audio_clips = [a.get_moviepy_clip() for a in self.audio_clips]
                final_audio = CompositeAudioClip(audio_clips)
                final_video = final_video.set_audio(final_audio)

            final_video.write_videofile(path, codec="libx264", audio_codec="aac",
                                        bitrate="8000k", fps=30, logger=None)
            final_video.close()
            messagebox.showinfo("导出完成", f"视频已保存到：\n{path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    # ========== 工具方法 ==========
    def _refresh_player(self):
        self.player.update_clips(self.video_clips, self.audio_clips)

    def _on_close(self):
        self.player.stop()
        for c in self.video_clips:
            if hasattr(c, 'release'):
                c.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEditorApp(root)
    root.mainloop()
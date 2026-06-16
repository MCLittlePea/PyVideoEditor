import tkinter as tk
from tkinter import filedialog, ttk, messagebox, colorchooser
import cv2
from PIL import Image, ImageTk
import numpy as np
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip,
    TextClip, concatenate_videoclips, vfx, afx
)
import os
import sys
import threading


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    elif hasattr(sys, '_NUITKA_ONEFILE_PARENT'):
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


ffmpeg_packed_path = get_resource_path("imageio/ffmpeg/ffmpeg-win64-v4.2.2.exe")
if os.path.exists(ffmpeg_packed_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_packed_path


class PyVideoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("PyVideoEditor - Python视频编辑器")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        self.video_path = None
        self.cap = None
        self.video_clip = None
        self.fps = 0
        self.total_frames = 0
        self.duration = 0
        self.current_frame = 0
        self.is_playing = False
        self.in_point = 0
        self.out_point = 0

        self.video_list = []

        self.export_format = "mp4"
        self.export_quality = "high"

        self.create_ui()

        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.seek_backward())
        self.root.bind("<Right>", lambda e: self.seek_forward())
        self.root.bind("<i>", lambda e: self.set_in_point())
        self.root.bind("<o>", lambda e: self.set_out_point())

    def create_ui(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, bg="#2b2b2b", width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(left_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_frame = tk.Frame(left_frame, bg="#2b2b2b")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.play_button = tk.Button(control_frame, text="▶ 播放", command=self.toggle_play, width=8)
        self.play_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = tk.Button(control_frame, text="⏹ 停止", command=self.stop_video, width=8)
        self.stop_button.pack(side=tk.LEFT, padx=2)

        self.timeline = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_timeline_change)
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.time_label = tk.Label(control_frame, text="00:00:00 / 00:00:00", bg="#2b2b2b", fg="white")
        self.time_label.pack(side=tk.LEFT, padx=5)

        self.in_button = tk.Button(control_frame, text="入点 [I]", command=self.set_in_point, width=6)
        self.in_button.pack(side=tk.LEFT, padx=2)

        self.out_button = tk.Button(control_frame, text="出点 [O]", command=self.set_out_point, width=6)
        self.out_button.pack(side=tk.LEFT, padx=2)

        right_frame = tk.Frame(main_frame, bg="#ffffff", width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        import_tab = ttk.Frame(self.notebook)
        self.notebook.add(import_tab, text="导入")

        tk.Button(import_tab, text="导入视频", command=self.import_video,
                  width=30, height=2).pack(pady=10, padx=10)

        tk.Label(import_tab, text="视频合并列表:").pack(anchor=tk.W, padx=10, pady=(10, 0))

        list_frame = tk.Frame(import_tab)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.merge_listbox = tk.Listbox(list_frame, width=35, height=8)
        self.merge_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.merge_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.merge_listbox.config(yscrollcommand=scrollbar.set)

        tk.Button(import_tab, text="添加到合并列表", command=self.add_to_merge).pack(pady=5)
        tk.Button(import_tab, text="从列表移除", command=self.remove_from_merge).pack(pady=5)
        tk.Button(import_tab, text="合并视频", command=self.merge_videos, bg="#4CAF50", fg="white").pack(pady=10)

        edit_tab = ttk.Frame(self.notebook)
        self.notebook.add(edit_tab, text="编辑")

        cut_frame = ttk.LabelFrame(edit_tab, text="视频剪切")
        cut_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(cut_frame, text="入点:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.in_entry = tk.Entry(cut_frame, width=10)
        self.in_entry.grid(row=0, column=1, padx=5, pady=5)
        self.in_entry.insert(0, "00:00:00")

        tk.Label(cut_frame, text="出点:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.out_entry = tk.Entry(cut_frame, width=10)
        self.out_entry.grid(row=1, column=1, padx=5, pady=5)
        self.out_entry.insert(0, "00:00:00")

        tk.Button(cut_frame, text="剪切并导出", command=self.cut_video,
                  bg="#2196F3", fg="white").grid(row=2, column=0, columnspan=2, pady=10)

        transform_frame = ttk.LabelFrame(edit_tab, text="视频变换")
        transform_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(transform_frame, text="顺时针旋转90°",
                  command=lambda: self.apply_transform("rotate90")).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(transform_frame, text="逆时针旋转90°",
                  command=lambda: self.apply_transform("rotate-90")).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(transform_frame, text="水平翻转",
                  command=lambda: self.apply_transform("hflip")).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(transform_frame, text="垂直翻转",
                  command=lambda: self.apply_transform("vflip")).pack(fill=tk.X, padx=5, pady=2)

        effect_tab = ttk.Frame(self.notebook)
        self.notebook.add(effect_tab, text="效果")

        color_frame = ttk.LabelFrame(effect_tab, text="颜色效果")
        color_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(color_frame, text="黑白效果",
                  command=lambda: self.apply_effect("blackwhite")).pack(fill=tk.X, padx=5, pady=2)

        tk.Label(color_frame, text="亮度调节:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.brightness_scale = ttk.Scale(color_frame, from_=0.1, to=2.0, value=1.0, orient=tk.HORIZONTAL)
        self.brightness_scale.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(color_frame, text="应用亮度",
                  command=lambda: self.apply_effect("brightness")).pack(fill=tk.X, padx=5, pady=2)

        speed_frame = ttk.LabelFrame(effect_tab, text="播放速度")
        speed_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(speed_frame, text="速度倍数:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.speed_scale = ttk.Scale(speed_frame, from_=0.25, to=4.0, value=1.0, orient=tk.HORIZONTAL)
        self.speed_scale.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(speed_frame, text="应用速度",
                  command=lambda: self.apply_effect("speed")).pack(fill=tk.X, padx=5, pady=2)

        audio_tab = ttk.Frame(self.notebook)
        self.notebook.add(audio_tab, text="音频")

        tk.Button(audio_tab, text="提取音频为MP3",
                  command=self.extract_audio).pack(fill=tk.X, padx=10, pady=10)

        volume_frame = ttk.LabelFrame(audio_tab, text="音量调节")
        volume_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(volume_frame, text="音量倍数:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.volume_scale = ttk.Scale(volume_frame, from_=0.0, to=2.0, value=1.0, orient=tk.HORIZONTAL)
        self.volume_scale.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(volume_frame, text="应用音量",
                  command=lambda: self.apply_effect("volume")).pack(fill=tk.X, padx=5, pady=2)

        tk.Button(audio_tab, text="替换音频",
                  command=self.replace_audio).pack(fill=tk.X, padx=10, pady=10)

        text_tab = ttk.Frame(self.notebook)
        self.notebook.add(text_tab, text="文本")

        tk.Label(text_tab, text="文本内容:").pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.text_content = tk.Entry(text_tab, width=30)
        self.text_content.pack(fill=tk.X, padx=10, pady=5)
        self.text_content.insert(0, "示例文本")

        tk.Label(text_tab, text="字体大小:").pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.font_size = tk.Scale(text_tab, from_=10, to=100, value=30, orient=tk.HORIZONTAL)
        self.font_size.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(text_tab, text="显示时长(秒):").pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.text_duration = tk.Entry(text_tab, width=10)
        self.text_duration.pack(fill=tk.X, padx=10, pady=5)
        self.text_duration.insert(0, "5")

        tk.Label(text_tab, text="位置:").pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.text_position = ttk.Combobox(text_tab, values=["顶部", "中间", "底部", "左上角", "右上角", "左下角", "右下角"])
        self.text_position.pack(fill=tk.X, padx=10, pady=5)
        self.text_position.current(2)

        self.text_color = "#FFFFFF"
        tk.Button(text_tab, text="选择颜色", command=self.choose_text_color).pack(fill=tk.X, padx=10, pady=5)

        tk.Button(text_tab, text="添加文本", command=self.add_text,
                  bg="#9C27B0", fg="white").pack(fill=tk.X, padx=10, pady=10)

        export_tab = ttk.Frame(self.notebook)
        self.notebook.add(export_tab, text="导出")

        tk.Label(export_tab, text="导出格式:").pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.format_combo = ttk.Combobox(export_tab, values=["mp4", "avi", "mov", "gif"])
        self.format_combo.pack(fill=tk.X, padx=10, pady=5)
        self.format_combo.current(0)

        tk.Label(export_tab, text="导出质量:").pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.quality_combo = ttk.Combobox(export_tab, values=["低", "中", "高", "极高"])
        self.quality_combo.pack(fill=tk.X, padx=10, pady=5)
        self.quality_combo.current(2)

        tk.Button(export_tab, text="导出视频", command=self.export_video,
                  bg="#FF5722", fg="white", height=2).pack(fill=tk.X, padx=10, pady=20)

        self.status_bar = tk.Label(self.root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_bar = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

    def import_video(self):
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("所有文件", "*.*")]
        )

        if file_path:
            self.video_path = file_path
            self.load_video()

    def load_video(self):
        try:
            if self.cap is not None:
                self.cap.release()
            if self.video_clip is not None:
                self.video_clip.close()

            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps

            self.video_clip = VideoFileClip(self.video_path)

            self.timeline.config(to=self.duration)
            self.current_frame = 0
            self.in_point = 0
            self.out_point = self.duration

            self.update_time_display()
            self.show_frame()

            self.status_bar.config(text=f"已加载: {os.path.basename(self.video_path)} | 时长: {self.format_time(self.duration)} | 分辨率: {self.video_clip.size[0]}x{self.video_clip.size[1]}")

        except Exception as e:
            messagebox.showerror("错误", f"加载视频失败: {str(e)}")

    def show_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            if label_width > 1 and label_height > 1:
                h, w = frame.shape[:2]
                ratio = min(label_width / w, label_height / h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                frame = cv2.resize(frame, (new_w, new_h))

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk

            self.timeline.set(self.current_frame / self.fps)
            self.update_time_display()

    def toggle_play(self):
        if self.cap is None:
            return

        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_button.config(text="⏸ 暂停")
            self.play_video()
        else:
            self.play_button.config(text="▶ 播放")

    def play_video(self):
        if self.is_playing and self.cap is not None:
            self.current_frame += 1

            if self.current_frame >= self.total_frames:
                self.current_frame = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.show_frame()

            delay = int(1000 / self.fps)
            self.root.after(delay, self.play_video)

    def stop_video(self):
        self.is_playing = False
        self.play_button.config(text="▶ 播放")
        self.current_frame = 0
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.show_frame()

    def seek_backward(self):
        if self.cap is None:
            return

        self.current_frame = max(0, self.current_frame - int(self.fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.show_frame()

    def seek_forward(self):
        if self.cap is None:
            return

        self.current_frame = min(self.total_frames - 1, self.current_frame + int(self.fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.show_frame()

    def on_timeline_change(self, value):
        if self.cap is None:
            return

        try:
            time_pos = float(value)
            self.current_frame = int(time_pos * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.show_frame()
        except:
            pass

    def set_in_point(self):
        if self.cap is None:
            return

        self.in_point = self.current_frame / self.fps
        self.in_entry.delete(0, tk.END)
        self.in_entry.insert(0, self.format_time(self.in_point))
        self.status_bar.config(text=f"入点设置为: {self.format_time(self.in_point)}")

    def set_out_point(self):
        if self.cap is None:
            return

        self.out_point = self.current_frame / self.fps
        self.out_entry.delete(0, tk.END)
        self.out_entry.insert(0, self.format_time(self.out_point))
        self.status_bar.config(text=f"出点设置为: {self.format_time(self.out_point)}")

    def update_time_display(self):
        current_time = self.current_frame / self.fps
        self.time_label.config(text=f"{self.format_time(current_time)} / {self.format_time(self.duration)}")

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def parse_time(self, time_str):
        try:
            parts = list(map(int, time_str.split(':')))
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
            else:
                return float(time_str)
        except:
            return 0

    def add_to_merge(self):
        if self.video_path:
            self.video_list.append(self.video_path)
            self.merge_listbox.insert(tk.END, os.path.basename(self.video_path))

    def remove_from_merge(self):
        selected = self.merge_listbox.curselection()
        if selected:
            index = selected[0]
            self.merge_listbox.delete(index)
            del self.video_list[index]

    def merge_videos(self):
        if len(self.video_list) < 2:
            messagebox.showwarning("警告", "请至少添加两个视频到合并列表")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )

        if not output_path:
            return

        threading.Thread(target=self._merge_videos_thread, args=(output_path,), daemon=True).start()

    def _merge_videos_thread(self, output_path):
        try:
            self.status_bar.config(text="正在合并视频...")
            self.progress_bar.start()

            clips = []
            for video_path in self.video_list:
                clip = VideoFileClip(video_path)
                clips.append(clip)

            final_clip = concatenate_videoclips(clips)

            bitrate = self.get_bitrate()

            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                bitrate=bitrate,
                verbose=False,
                logger=None
            )

            for clip in clips:
                clip.close()
            final_clip.close()

            self.progress_bar.stop()
            self.status_bar.config(text="视频合并完成!")
            messagebox.showinfo("成功", "视频合并完成!")

        except Exception as e:
            self.progress_bar.stop()
            self.status_bar.config(text="合并失败")
            messagebox.showerror("错误", f"合并视频失败: {str(e)}")

    def cut_video(self):
        if self.video_clip is None:
            messagebox.showwarning("警告", "请先导入视频")
            return

        in_time = self.parse_time(self.in_entry.get())
        out_time = self.parse_time(self.out_entry.get())

        if in_time >= out_time or out_time > self.duration:
            messagebox.showerror("错误", "无效的时间范围")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )

        if not output_path:
            return

        threading.Thread(target=self._cut_video_thread, args=(output_path, in_time, out_time), daemon=True).start()

    def _cut_video_thread(self, output_path, in_time, out_time):
        try:
            self.status_bar.config(text="正在剪切视频...")
            self.progress_bar.start()

            cut_clip = self.video_clip.subclip(in_time, out_time)

            bitrate = self.get_bitrate()

            cut_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                bitrate=bitrate,
                verbose=False,
                logger=None
            )

            cut_clip.close()

            self.progress_bar.stop()
            self.status_bar.config(text="视频剪切完成!")
            messagebox.showinfo("成功", "视频剪切完成!")

        except Exception as e:
            self.progress_bar.stop()
            self.status_bar.config(text="剪切失败")
            messagebox.showerror("错误", f"剪切视频失败: {str(e)}")

    def apply_transform(self, transform_type):
        if self.video_clip is None:
            messagebox.showwarning("警告", "请先导入视频")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )

        if not output_path:
            return

        threading.Thread(target=self._apply_transform_thread, args=(output_path, transform_type), daemon=True).start()

    def _apply_transform_thread(self, output_path, transform_type):
        try:
            self.status_bar.config(text=f"正在应用{transform_type}变换...")
            self.progress_bar.start()

            if transform_type == "rotate90":
                transformed_clip = self.video_clip.rotate(90)
            elif transform_type == "rotate-90":
                transformed_clip = self.video_clip.rotate(-90)
            elif transform_type == "hflip":
                transformed_clip = self.video_clip.fx(vfx.mirror_x)
            elif transform_type == "vflip":
                transformed_clip = self.video_clip.fx(vfx.mirror_y)

            bitrate = self.get_bitrate()

            transformed_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                bitrate=bitrate,
                verbose=False,
                logger=None
            )

            transformed_clip.close()

            self.progress_bar.stop()
            self.status_bar.config(text="变换应用完成!")
            messagebox.showinfo("成功", "视频变换完成!")

        except Exception as e:
            self.progress_bar.stop()
            self.status_bar.config(text="变换失败")
            messagebox.showerror("错误", f"应用变换失败: {str(e)}")

    def apply_effect(self, effect_type):
        if self.video_clip is None:
            messagebox.showwarning("警告", "请先导入视频")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )

        if not output_path:
            return

        threading.Thread(target=self._apply_effect_thread, args=(output_path, effect_type), daemon=True).start()

    def _apply_effect_thread(self, output_path, effect_type):
        try:
            self.status_bar.config(text=f"正在应用{effect_type}效果...")
            self.progress_bar.start()

            if effect_type == "blackwhite":
                effect_clip = self.video_clip.fx(vfx.blackwhite)
            elif effect_type == "brightness":
                brightness = self.brightness_scale.get()
                effect_clip = self.video_clip.fx(vfx.colorx, brightness)
            elif effect_type == "speed":
                speed = self.speed_scale.get()
                effect_clip = self.video_clip.fx(vfx.speedx, speed)
            elif effect_type == "volume":
                volume = self.volume_scale.get()
                effect_clip = self.video_clip.fx(afx.volumex, volume)

            bitrate = self.get_bitrate()

            effect_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                bitrate=bitrate,
                verbose=False,
                logger=None
            )

            effect_clip.close()

            self.progress_bar.stop()
            self.status_bar.config(text="效果应用完成!")
            messagebox.showinfo("成功", "视频效果应用完成!")

        except Exception as e:
            self.progress_bar.stop()
            self.status_bar.config(text="效果应用失败")
            messagebox.showerror("错误", f"应用效果失败: {str(e)}")

    def extract_audio(self):
        if self.video_clip is None:
            messagebox.showwarning("警告", "请先导入视频")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3文件", "*.mp3"), ("所有文件", "*.*")]
        )

        if not output_path:
            return

        threading.Thread(target=self._extract_audio_thread, args=(output_path,), daemon=True).start()

    def _extract_audio_thread(self, output_path):
        try:
            self.status_bar.config(text="正在提取音频...")
            self.progress_bar.start()

            audio = self.video_clip.audio
            audio.write_audiofile(
                output_path,
                codec="mp3",
                verbose=False,
                logger=None
            )

            audio.close()

            self.progress_bar.stop()
            self.status_bar.config(text="音频提取完成!")
            messagebox.showinfo("成功", "音频提取完成!")

        except Exception as e:
            self.progress_bar.stop()
            self.status_bar.config(text="音频提取失败")
            messagebox.showerror("错误", f"提取音频失败: {str(e)}")

    def replace_audio(self):
        if self.video_clip is None:
            messagebox.showwarning("警告", "请先导入视频")
            return

        audio_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.mp3 *.wav *.flac *.m4a"), ("所有文件", "*.*")]
        )

        if not audio_path:
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )

        if not output_path:
            return

        threading.Thread(target=self._replace_audio_thread, args=(output_path, audio_path), daemon=True).start()

    def _replace_audio_thread(self, output_path, audio_path):
        try:
            self.status_bar.config(text="正在替换音频...")
            self.progress_bar.start()

            new_audio = AudioFileClip(audio_path)

            if new_audio.duration > self.video_clip.duration:
                new_audio = new_audio.subclip(0, self.video_clip.duration)

            final_clip = self.video_clip.set_audio(new_audio)

            bitrate = self.get_bitrate()

            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                bitrate=bitrate,
                verbose=False,
                logger=None
            )

            new_audio.close()
            final_clip.close()

            self.progress_bar.stop()
            self.status_bar.config(text="音频替换完成!")
            messagebox.showinfo("成功", "音频替换完成!")

        except Exception as e:
            self.progress_bar.stop()
            self.status_bar.config(text="音频替换失败")
            messagebox.showerror("错误", f"替换音频失败: {str(e)}")

    def choose_text_color(self):
        color = colorchooser.askcolor(title="选择文本颜色")
        if color[1]:
            self.text_color = color[1]

    def add_text(self):
        if self.video_clip is None:
            messagebox.showwarning("警告", "请先导入视频")
            return

        text = self.text_content.get()
        if not text:
            messagebox.showwarning("警告", "请输入文本内容")
            return

        try:
            duration = float(self.text_duration.get())
        except:
            messagebox.showerror("错误", "无效的时长")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("所有文件", "*.*")]
        )

        if not output_path:
            return

        threading.Thread(target=self._add_text_thread, args=(output_path, text, duration), daemon=True).start()

    def _add_text_thread(self, output_path, text, duration):
        try:
            self.status_bar.config(text="正在添加文本...")
            self.progress_bar.start()

            font_path = "C:/Windows/Fonts/simhei.ttf"
            if not os.path.exists(font_path):
                font_path = get_resource_path("fonts/simhei.ttf")

            txt_clip = TextClip(
                text,
                fontsize=self.font_size.get(),
                color=self.text_color,
                font=font_path
            )

            position_map = {
                "顶部": ("center", "top"),
                "中间": ("center", "center"),
                "底部": ("center", "bottom"),
                "左上角": ("left", "top"),
                "右上角": ("right", "top"),
                "左下角": ("left", "bottom"),
                "右下角": ("right", "bottom")
            }

            pos = position_map.get(self.text_position.get(), ("center", "bottom"))
            txt_clip = txt_clip.set_position(pos).set_duration(min(duration, self.video_clip.duration))

            final_clip = CompositeVideoClip([self.video_clip, txt_clip])

            bitrate = self.get_bitrate()

            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                bitrate=bitrate,
                verbose=False,
                logger=None
            )

            txt_clip.close()
            final_clip.close()

            self.progress_bar.stop()
            self.status_bar.config(text="文本添加完成!")
            messagebox.showinfo("成功", "文本添加完成!")

        except Exception as e:
            self.progress_bar.stop()
            self.status_bar.config(text="文本添加失败")
            messagebox.showerror("错误", f"添加文本失败: {str(e)}")

    def export_video(self):
        if self.video_clip is None:
            messagebox.showwarning("警告", "请先导入视频")
            return

        format_ext = self.format_combo.get()
        output_path = filedialog.asksaveasfilename(
            defaultextension=f".{format_ext}",
            filetypes=[(f"{format_ext.upper()}文件", f"*.{format_ext}"), ("所有文件", "*.*")]
        )

        if not output_path:
            return

        threading.Thread(target=self._export_video_thread, args=(output_path, format_ext), daemon=True).start()

    def _export_video_thread(self, output_path, format_ext):
        try:
            self.status_bar.config(text="正在导出视频...")
            self.progress_bar.start()

            bitrate = self.get_bitrate()

            if format_ext == "gif":
                self.video_clip.write_gif(
                    output_path,
                    fps=15,
                    verbose=False,
                    logger=None
                )
            else:
                codec_map = {
                    "mp4": "libx264",
                    "avi": "mpeg4",
                    "mov": "libx264"
                }

                codec = codec_map.get(format_ext, "libx264")

                self.video_clip.write_videofile(
                    output_path,
                    codec=codec,
                    audio_codec="aac",
                    bitrate=bitrate,
                    verbose=False,
                    logger=None
                )

            self.progress_bar.stop()
            self.status_bar.config(text="视频导出完成!")
            messagebox.showinfo("成功", "视频导出完成!")

        except Exception as e:
            self.progress_bar.stop()
            self.status_bar.config(text="导出失败")
            messagebox.showerror("错误", f"导出视频失败: {str(e)}")

    def get_bitrate(self):
        quality_map = {
            "低": "1000k",
            "中": "3000k",
            "高": "5000k",
            "极高": "10000k"
        }
        return quality_map.get(self.quality_combo.get(), "5000k")

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        if self.video_clip is not None:
            self.video_clip.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PyVideoEditor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
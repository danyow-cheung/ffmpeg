import os 
import ffmpeg # pip install ffmpeg-python
import numpy as np 
import cv2 
import librosa


class VideoWriter(object):
    def __init__(self, video_write_path: str,
                 height: int, width: int, fps: float,
                 pixel_format: str = "rgb24",
                 audio: any = None) -> None:
        """
        Params:
            video_write_path(str): 视频文件保存路径;
            height(int): 输出视频尺寸的高;
            width(int):  输出视频尺寸的宽;
            fps(int): 输出视频帧率;
            pixel_format(str): ["rgb24", "rgba"];
            audio(any): 输出音频;
        """
        self.frame_count = 0
        self.frame_height = height
        self.frame_width = width
        self.frame_per_second = fps
        self.input_pix_fmt = pixel_format

        if pixel_format == "rgba":
            if audio is not None:
                self.writer_process = (
                    ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt=self.input_pix_fmt,
                           s="{}x{}".format(width, height), framerate=fps)
                    .output(audio, video_write_path,
                            vcodec="libvpx", pix_fmt="yuva420p",
                            acodec="libvorbis", loglevel="error",
                            )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
            else:
                self.writer_process = (
                    ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt=self.input_pix_fmt,
                           s="{}x{}".format(width, height), framerate=fps)
                    .output(video_write_path,
                            vcodec="libvpx", pix_fmt="yuva420p",
                            loglevel="error",
                            )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
        elif pixel_format == "rgb24":
            if audio is not None:
                self.writer_process = (
                    ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt=self.input_pix_fmt,
                           s="{}x{}".format(width, height), framerate=fps)
                    .output(audio, video_write_path,
                            vcodec="libx264", pix_fmt="yuv420p",
                            acodec="aac", loglevel="error",
                            )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
            else:
                self.writer_process = (
                    ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt=self.input_pix_fmt,
                           s="{}x{}".format(width, height), framerate=fps)
                    .output(video_write_path,
                            vcodec="libx264", pix_fmt="yuv420p",
                            loglevel="error",
                            )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
        else:
            raise ValueError(f"{pixel_format} is not support for now.")

    def __len__(self):
        return self.frame_count

    def write(self, image: np.ndarray):
        """
        Params:
            image(np.ndarray): 输入视频帧, 请确保高宽与初始化时的高宽一致,
                色彩空间为`RGB`或`RGBA`;
        """
        assert image.shape[0] == self.frame_height and image.shape[1] == self.frame_width
        if self.input_pix_fmt == "rgb24":
            assert image.shape[2] == 3
        else:
            assert image.shape[2] == 4

        frame = image.astype("uint8").tobytes()
        self.writer_process.stdin.write(frame)
        self.frame_count += 1

    def close(self):
        self.writer_process.stdin.close()
        self.writer_process.wait()


class VideoReader(object):
    def __init__(self, video_read_path: str) -> None:
        """
        Params:
            video_read_path(str): 视频文件路径;
        """

        self.cap = cv2.VideoCapture(video_read_path)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_conut = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_per_second = round(self.cap.get(cv2.CAP_PROP_FPS), 2)
        self.current_frame_index = 0

    def __len__(self):
        return self.frame_conut

    @property
    def shape(self):
        return (self.frame_height, self.frame_width)

    @property
    def fps(self):
        return self.frame_per_second

    def __getitem__(self, idx):
        if idx == self.current_frame_index:
            still_reading, frame = self.cap.read()
            self.current_frame_index += 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            still_reading, frame = self.cap.read()
            self.current_frame_index = idx + 1

        if not still_reading:
            raise ValueError(f"Video Read Error, at frame index: {idx}")

        return frame

    def close(self):
        self.cap.release()
def get_video_seconds(video_path: str):
    """获取视频的时长.

    Params:
        video_path(str): 视频路径.

    Returns:
        float: 视频的时长, 单位为秒.
    """
    assert os.path.exists(video_path), \
        "The input video path does not exist."

    video_reader = VideoReader(video_path)
    video_seconds = float(len(video_reader) / video_reader.fps)
    video_reader.close()

    del video_reader

    return video_seconds


def cut_video(video_file:str,start_seconds:int,end_seconds:int,output_prefix:str):
    '''
    裁剪视频-无声
    '''
    video = VideoReader(video_file)
    fps = video.frame_per_second    
    # 图像开始/结束的索引
    start_idx = int(start_seconds * fps )
    end_idx = int(end_seconds * fps )
    
    writer = VideoWriter(
        video_write_path=r'Dataset\huang_video_simplecut\test.mp4',
        height=video.frame_height,
        width=video.frame_height,
        fps=fps,
        pixel_format='rgb24',
    )

    for count in range(start_idx,end_idx+1):
        image = video.__getitem__(count)
    
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # 通常情况下，视频编解码器期望图像的通道顺序为 BGR，而不是常见的 RGB 顺序。
        writer.write(bgr_image)
    print('done')

def cut_video_with_audio(video_file:str,start_seconds:int,end_seconds:int):
    video = VideoReader(video_file)


    audio, sr = librosa.load(video_file, sr=None)
    fps=video.frame_per_second
    start_sample = int(start_seconds * sr)
    end_sample = int(end_seconds * sr)
    start_idx = int(start_seconds * fps )
    end_idx = int(end_seconds * fps )

    # 截取音频片段
    segment = audio[start_sample:end_sample]
    output_file= r'Dataset\huang_video_simplecut\test.wav'
    # librosa.write(output_file, segment, sr)
    # from scipy.io import wavefile 
    from scipy.io.wavfile import write
    write(output_file,sr,segment)
    

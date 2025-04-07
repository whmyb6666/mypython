# 把视频与音频文件进行合并

from moviepy.editor import *

# 修改视频的长宽比
def changeResize():
    # do something ...
    # print(tm.getTimeSec())
    video = VideoFileClip('runs/detect/2024-07-05 18_21_32_audio.mp4')
    width,height=video.size
    new_width,new_height=width*2,height*2
    #
    new_video= video.resize((new_width,new_height))
    new_video.write_videofile('runs/detect/2024-07-05 18_21_32_audio_new.mp4',bitrate='5000k')


# 设置音频和视频的时间偏移，确保音频在视频的第3秒开始播放
def megreMP4andAudio(mp_filename,mp4_audio_filename):
    audioclip = AudioFileClip(WAVE_OUTPUT_FILENAME)
    videoclip = VideoFileClip(mp_filename)

    # 设置音频和视频的时间偏移，确保音频在视频的第3秒开始播放
    audio_delay = 5

    # 应用时间偏移到音频
    offset_audio_clip = audioclip.set_start(audio_delay)

    # 将视频和音频合并
    final_clip = videoclip.set_audio(offset_audio_clip)

    #print(videoclip)
    #videoclip2 = videoclip.set_audio(audioclip)
    #video = CompositeVideoClip([videoclip2])
    video = CompositeVideoClip([final_clip])
    video.write_videofile(mp4_audio_filename, codec='mpeg4',bitrate='5000k')


def mergeMP4andAudio2(mp4_filename,wav_filename,mp4_audio_filename):
    audioclip = AudioFileClip(wav_filename)
    videoclip = VideoFileClip(mp4_filename)
    print(videoclip)
    # 设置音频和视频的时间偏移，确保音频在视频的第3秒开始播放
    #audio_delay = 5
    # 应用时间偏移到音频
    #offset_audio_clip = audioclip.set_start(audio_delay)
    #final_clip = videoclip.set_audio(offset_audio_clip)
    #final_clip = videoclip.set_audio(audioclip)
    #video = CompositeVideoClip([final_clip])
    # 将音频的时长调整为与视频一致
    chang_rate = audioclip.duration/videoclip.duration

    new_audio = audioclip.fl_time(lambda t: chang_rate * t, apply_to=['mask', 'audio'])
    new_audio = new_audio.set_duration(audioclip.duration /chang_rate)
    print(videoclip.duration,audioclip.duration,new_audio.duration)
    # 将音频和视频进行合并
    final_clip = videoclip.set_audio(new_audio)
    video = CompositeVideoClip([final_clip])
    video.write_videofile(mp4_audio_filename, codec='mpeg4',bitrate='5000k')

filename = '2024-07-11 10_27_37'
WAVE_OUTPUT_FILENAME = f"runs/detect/audio/{filename}.wav"
mp_filename = f'runs/detect/{filename}.mp4'
mp4_audio_filename= f'runs/detect/{filename}_audio_new_test.mp4'


mergeMP4andAudio2(mp_filename,WAVE_OUTPUT_FILENAME,mp4_audio_filename)


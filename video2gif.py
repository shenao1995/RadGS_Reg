from moviepy.editor import *


def mp4_to_gif(input_path, output_path, fps=10, scale=None):
    """
    将 MP4 文件转换为 GIF
    :param input_path: 输入的 MP4 文件路径
    :param output_path: 输出的 GIF 文件路径
    :param fps: 帧率（默认 10）
    :param scale: 缩放比例（例如 0.5 表示缩小一半，None 表示不缩放）
    """
    # 加载 MP4 文件
    clip = VideoFileClip(input_path)

    # 可选：调整大小
    if scale is not None:
        clip = clip.resize(scale)

    # 写入 GIF 文件
    clip.write_gif(output_path, fps=fps)

    print(f"GIF 已保存到 {output_path}")


# 示例用法
if __name__ == "__main__":
    input_mp4 = "assets/reg_results.mp4"  # 替换为你的 MP4 文件路径
    output_gif = "assets/reg_results.gif"  # 输出的 GIF 文件路径

    # 可选参数：调整帧率或缩放
    mp4_to_gif(input_mp4, output_gif, fps=10, scale=0.5)
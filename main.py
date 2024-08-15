from pathlib import Path

from audio_separator.separator import Separator


def main():
    input_audio = "/home/geek/workspace/TestSample/Tom_Holland.wav"
    output_dir = "/home/geek/workspace/TestSample/"
    model_file_dir = "/home/geek/.cache/audio-separator-models"
    vocal_separation_model_filename = "UVR-MDX-NET-Voc_FT.onnx"
    de_reverb_model_filename = "5_HP-Karaoke-UVR.pth"
    output_format = "wav"

    vocal_separation_separator = Separator(
        model_file_dir=model_file_dir,
        output_format=output_format,
        output_dir=output_dir,
        output_single_stem="Vocals",
        sample_rate=44100,
        mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 16, "enable_denoise": False},
        vr_params={"batch_size": 16, "window_size": 320, "aggression": 10, "enable_tta": True,
                   "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
    )

    vocal_separation_separator.load_model(model_filename=vocal_separation_model_filename)

    vocal_separation_output_file_path = vocal_separation_separator.separate(input_audio)[0]
    print(vocal_separation_output_file_path)

    vocal_separation_separator.load_model(model_filename=de_reverb_model_filename)

    de_reverb_output_file_path = \
    vocal_separation_separator.separate(Path(output_dir) / vocal_separation_output_file_path)[0]
    print(de_reverb_output_file_path)


if __name__ == "__main__":
    main()

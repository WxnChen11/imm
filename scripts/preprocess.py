import cv2
import os 
import random
from pathlib import Path
import progressbar

def split_video(video_path, output_dir, filename, frame_delay, n_pairs):
    # Given a video, generate n pairs of images with a frame delay between then
    # Save the images under filename.

    all_files = []

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if length <= frame_delay:
        return

    for i in range(n_pairs):

        starting = random.randrange(length - frame_delay - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, starting)
        res, frame = cap.read()
        cv2.imwrite(os.path.join(output_dir, filename + '_' + str(i) + '_0.png'),frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, starting + frame_delay)
        res, frame = cap.read()

        cv2.imwrite(os.path.join(output_dir, filename + '_' + str(i) + '_1.png'),frame)

        all_files.append(filename + '_' + str(i))

    cap.release()

    f = open(os.path.join(output_dir, "bfast_actions_pair_names.txt"), "a")
    f.write("\n".join(all_files))
    f.close()

def main(args):

    all_videos = []
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    kitchen_folders = [ f for f in os.scandir(args.video_dir) if f.is_dir() ]
    
    for kitchen_file in progressbar.progressbar(kitchen_folders):
        kitchen = kitchen_file.name

        camera_types_folders = [ f for f in os.scandir(kitchen_file.path) if f.is_dir() ]

        for camera_type_file in camera_types_folders:
            camera_type = camera_type_file.name

            videos = [ f for f in os.scandir(camera_type_file.path) if f.name.endswith('.avi') ]

            # print(kitchen, camera_type, videos)

            all_videos += videos

            for video_file in videos:
                try:
                    split_video(video_file.path, args.output_dir, video_file.name.split('.')[0] + '_' + camera_type, args.pair_delay_frames, args.n_pairs)
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process Video into Frame Pairs')
    parser.add_argument('--video_dir',type=str,default=None,
                        help='directory containing videos')
    parser.add_argument('--pair_delay_frames',type=int,default=20,
                        help='the number of frames between extracted pairs')
    parser.add_argument('--output_dir',type=str,default=None,
                        help='the directory to output data')
    parser.add_argument('--n_pairs',type=int,default=5,
                        help='the number of pairs to generate per video')
    parser.add_argument('--camera_type',type=str,default=None,
                        help='camera type to use. None uses all.')
    args = parser.parse_args()
    main(args)

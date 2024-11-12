import json
import os
import pandas as pd
from tqdm import tqdm

def reduce_frames():
    """
    The purpose of this script is to reduce the number of frames in the dataset based on a given ratio (DEFAULT: 1:20)
    To do this the script will:
    1. Create a new folder for each recipe in the dataset called "frames_reduced"
    2. Copy the first frame of the recipe to the new folder
    3. Copy every nth frame to the new folder, based on the ratio
    4. Create a csv file that maps the old frame names to the new frame names
    5. Update the recipe.json file so that the frames in the actions are updated to the new frame names
    """

    RECIPES_PATH = "/your/data/path/tasty_dataset/ALL_RECIPES_without_videos"

    RATIO = 20


    stats = {
        "total_recipes": 0,
        "total_frames": 0,
        "total_frames_reduced": 0,
    }

    # for each folder in the recipes path
    for folder in tqdm(os.listdir(RECIPES_PATH)):
        if os.path.exists(f"{RECIPES_PATH}/{folder}/recipe.json") and os.path.exists(f"{RECIPES_PATH}/{folder}/frames"):
            frame_files = [f for f in os.listdir(f"{RECIPES_PATH}/{folder}/frames") if f.endswith(".jpg")]
            frame_files.sort()

            if os.path.exists(f"{RECIPES_PATH}/{folder}/frames_reduced"):
                # delete all files in the frames_reduced folder
                os.system(f"rm -rf {RECIPES_PATH}/{folder}/frames_reduced/*")
            else:
                # create a new folder for the reduced frames
                os.makedirs(f"{RECIPES_PATH}/{folder}/frames_reduced", exist_ok=True)

            # copy the first frame to the new folder
            os.system(f"cp {RECIPES_PATH}/{folder}/frames/{frame_files[0]} {RECIPES_PATH}/{folder}/frames_reduced/{frame_files[0]}")

            # Map every frame to the new frame name, with frames inbetween being mapped to the closest frame
            frame_mapping = {}
            for i in range(len(frame_files)):
                frame_number = i // RATIO
                frame_mapping[frame_files[i]] = f"{frame_number:05d}.jpg"
                if i % RATIO == 0:
                    os.system(f"cp {RECIPES_PATH}/{folder}/frames/{frame_files[i]} {RECIPES_PATH}/{folder}/frames_reduced/{frame_number:05d}.jpg")


            assert len(frame_mapping) == len(frame_files), f"{len(frame_mapping)} {len(set(frame_mapping.values()))} {len(frame_files)}"


            # create a csv file with the frame mapping
            df = pd.DataFrame(list(frame_mapping.items()), columns=["old_frame", "new_frame"])
            df.to_csv(f"{RECIPES_PATH}/{folder}/frame_reduction_mapping.csv", index=False)

            # update the recipe.json file
            with open(f"{RECIPES_PATH}/{folder}/recipe.json", "r") as f:
                recipe = json.load(f)

                for instruction in recipe["recipe"]["instructions"]:
                    for action in instruction["actions"]:
                        action["framesPath"] = os.path.join(RECIPES_PATH, folder, "frames_reduced")
                        if "startFrame" in action and action["startFrame"] is not None:
                            action["startFrame"] = int(frame_mapping[f"{action['startFrame']:05d}.jpg"].replace(".jpg", ""))
                        if "middleFrame" in action and action["middleFrame"] is not None:
                            action["middleFrame"] = int(frame_mapping[f"{action['middleFrame']:05d}.jpg"].replace(".jpg", ""))
                        if "endFrame" in action and action["endFrame"] is not None:
                            action["endFrame"] = int(frame_mapping[f"{action['endFrame']:05d}.jpg"].replace(".jpg", ""))

                        assert os.path.exists(f"{RECIPES_PATH}/{folder}/frames_reduced/{action['startFrame']:05d}.jpg"), f"{RECIPES_PATH}/{folder}/frames_reduced/{action['startFrame']:05d}.jpg"

            with open(f"{RECIPES_PATH}/{folder}/recipe.json", "w") as f:
                json.dump(recipe, f, indent=4)

            stats["total_recipes"] += 1
            stats["total_frames"] += len(frame_files)
            stats["total_frames_reduced"] += len(os.listdir(f"{RECIPES_PATH}/{folder}/frames_reduced"))


    print(f"Total recipes: {stats['total_recipes']}")
    print(f"Total frames: {stats['total_frames']}, Average frames per recipe: {stats['total_frames']/stats['total_recipes']}")
    print(f"Total frames reduced: {stats['total_frames_reduced']}, Average frames reduced per recipe: {stats['total_frames_reduced']/stats['total_recipes']}")


def fix_frames_path():
    RECIPES_PATH = "/your/data/path/tasty_dataset/ALL_RECIPES_without_videos"

    for folder in tqdm(os.listdir(RECIPES_PATH)):
        if os.path.exists(f"{RECIPES_PATH}/{folder}/recipe.json"):
            with open(f"{RECIPES_PATH}/{folder}/recipe.json", "r") as f:
                recipe = json.load(f)

            for instruction in recipe["recipe"]["instructions"]:
                for action in instruction["actions"]:
                    action["framesPath"] = os.path.join(RECIPES_PATH, folder, "frames_reduced")

                    if action["startFrame"] is not None:
                        assert os.path.exists(f"{RECIPES_PATH}/{folder}/frames_reduced/{action['startFrame']:05d}.jpg"), f"{RECIPES_PATH}/{folder}/frames_reduced/{action['startFrame']:05d}.jpg"
                    if action["middleFrame"] is not None:
                        assert os.path.exists(f"{RECIPES_PATH}/{folder}/frames_reduced/{action['middleFrame']:05d}.jpg"), f"{RECIPES_PATH}/{folder}/frames_reduced/{action['middleFrame']:05d}.jpg"
                    if action["endFrame"] is not None:
                        assert os.path.exists(f"{RECIPES_PATH}/{folder}/frames_reduced/{action['endFrame']:05d}.jpg"), f"{RECIPES_PATH}/{folder}/frames_reduced/{action['endFrame']:05d}.jpg"

            with open(f"{RECIPES_PATH}/{folder}/recipe.json", "w") as f:
                json.dump(recipe, f, indent=4)



if __name__ == "__main__":
    fix_frames_path()



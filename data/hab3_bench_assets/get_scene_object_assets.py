# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from typing import Callable, List, Dict, Optional
import shutil

def file_endswith(filepath:str, end_str:str)->bool:
    """
    Return whether or not the file ends with a string.
    """
    return filepath.endswith(end_str)

def find_files(root_dir: str, discriminator: Callable[[str, str], bool], disc_str:str) -> List[str]:
    """
    Recursively find all filepaths under a root directory satisfying a particular constraint as defined by a discriminator function.

    :param root_dir: The roor directory for the recursive search.
    :param discriminator: The discriminator function which takes a filepath and discriminator string and returns a bool.

    :return: The list of all absolute filepaths found satisfying the discriminator.
    """
    filepaths: List[str] = []

    if not os.path.exists(root_dir):
        print(" Directory does not exist: " + str(dir))
        return filepaths

    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            sub_dir_filepaths = find_files(entry_path, discriminator, disc_str)
            filepaths.extend(sub_dir_filepaths)
        # apply a user-provided discriminator function to cull filepaths
        elif discriminator(entry_path, disc_str):
            filepaths.append(entry_path)
    return filepaths

def get_model_ids_from_scene_instance_json(filepath: str) -> List[str]:
    """
    Scrape a list of all unique model ids from the scene instance file.
    """
    assert filepath.endswith(".scene_instance.json"), "Must be a scene instance JSON."

    model_ids = []

    with open(filepath, "r") as f:
        scene_conf = json.load(f)
        if "object_instances" in scene_conf:
            for obj_inst in scene_conf["object_instances"]:
                model_ids.append(obj_inst["template_name"])
        else:
            print("No object instances field detected, are you sure this is scene instance file?")

    print(f" {filepath} has {len(model_ids)} object instances.")
    model_ids = list(set(model_ids))
    print(f" {filepath} has {len(model_ids)} unique objects.")

    return model_ids

def copy_file_to(root_dir, source_file, destination_dir)->None:
    """
    Copied files with their relative folder structure to a target directory.
    """
    abs_dest = os.path.abspath(destination_dir)
    rel_source = os.path.relpath(source_file, root_dir)
    dest_rel_path = os.path.join(abs_dest, rel_source)
    print(f"source_file = {source_file}")
    print(f"dest_file = {dest_rel_path}")
    os.makedirs(os.path.dirname(dest_rel_path), exist_ok=True)
    shutil.copyfile(source_file, dest_rel_path)


#------------------------------------------------------
# Run this script to copy asset files for a scene into another directory (e.g. benchmark files)
# e.g. python get_scene_object_assets.py --dataset-root-dir <path-to>/fphab/ --scenes 102816009 --file-ends .glb .object_config.json .ply --dest-dir <path-to>/hab3_bench_assets/hab3-hssd/
#------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Get all specified asset files associated with the models in a given scene."
    )
    parser.add_argument(
        "--dataset-root-dir",
        type=str,
        help="path to HSSD SceneDataset root directory containing 'fphab-uncluttered.scene_dataset_config.json'.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        help="one or more scene ids",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=None,
        help="Path to destination directory if copy is desired.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="If selected, only render glbs will be selected (i.e., no collision or receptacle assets).",
    )
    parser.add_argument(
        "--file-ends",
        type=str,
        nargs="+",
        help="One or more file ending strings to look for."
    )

    args = parser.parse_args()
    scene_ids = list(dict.fromkeys(args.scenes))

    fp_root_dir = args.dataset_root_dir
    config_root_dir = os.path.join(fp_root_dir, "scenes-uncluttered")
    configs = find_files(config_root_dir, file_endswith, ".scene_instance.json")
    my_files = [f for end_str in args.file_ends for f in find_files(fp_root_dir, file_endswith, end_str)]
    
    #for render only
    if args.render_only:
        render_glbs = [f for f in my_files if (".collider" not in f and ".filteredSupportSurface" not in f and f.endswith(".glb"))]
        my_files = render_glbs

    scene_asset_filepaths = {}
    for filepath in configs:
        #these should be removed, but screen them for now
        if "orig" in filepath:
            print(f"Skipping alleged 'original' instance file {filepath}")
            continue
        for scene_id in scene_ids:
            #NOTE: add the extension back here to avoid partial matches
            if scene_id+".scene_instance.json" in filepath:
                print(f"filepath '{filepath}' matches scene_id '{scene_id}'")
                assert scene_id not in scene_asset_filepaths, f"Duplicate scene instance file {filepath} found for scene_id {scene_id}"
                scene_asset_filepaths[scene_id] = []
                model_ids = get_model_ids_from_scene_instance_json(filepath)
                for model_id in model_ids:
                    for f in my_files:
                        model_id_split = f.split(model_id)
                        #we only want files which exactly match the model id. Some short model ids are concatenated with another via '_', these must be culled.
                        if len(model_id_split) > 1 and model_id_split[-1][0] == "." and model_id_split[-2][-1] != "_":
                            if "part" in f and "part" not in model_id:
                                continue
                            if f not in scene_asset_filepaths[scene_id]:
                                scene_asset_filepaths[scene_id].append(f)


    for scene_id in scene_asset_filepaths.keys():
        print(f"Scene {scene_id}")
        for asset_path in scene_asset_filepaths[scene_id]:
            print(f"    {asset_path}")
            if args.dest_dir is not None:
                copy_file_to(fp_root_dir, asset_path, args.dest_dir)
    for scene, models in scene_asset_filepaths.items():
        print(f"    Scene {scene} contains {len(models)} requested assets.")


if __name__ == "__main__":
    main()
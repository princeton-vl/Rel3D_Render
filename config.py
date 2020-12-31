from pathlib import Path

__cur_dir = Path(__file__).resolve().parent
BLENDER_ENV = {
    'PATH': f"{__cur_dir}/blender-2.81-linux-glibc217-x86_64"
}
SHAPE_ROOT = f"{__cur_dir}/data/shapes"
BACKGROUND_WALL = f"{__cur_dir}/data/Playground.obj"
BACKGROUND_WALL_NO_FRONT = f"{__cur_dir}/data/Playground_without_front_wall.obj"
TASK_CSV = f"{__cur_dir}/data/assignments-contrastive-v1.csv,{__cur_dir}/data/assignments-contrastive.csv"
CACHE_DIR = f"{__cur_dir}/.cache"
TMP_DIR = f"{__cur_dir}/.tmp"
JSON_DATA = f"{__cur_dir}/data/full.json"

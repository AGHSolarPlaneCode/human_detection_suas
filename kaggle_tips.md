# kaggle

### Useful tips for model training

-  enable GPU in notebook settings
- move dataset from input to working directory for faster access 
```python
import shutil

source_dir = "/kaggle/input/datasets/bartoszkrolik/nomad-sliced-yolo-512/sliced_data"
dest_dir = "/kaggle/working/sliced_data"

shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
```

- use save and commit -> it works in the backgorund and saves results, limit 12h
- use this double gpu and send in model params device=[0,1] so it uses both than you have 30gb of vram
- cache probably will fail on kaggle, because you overuse ram
# Directory Structure Report: `whole_nomad`

**Path:** `C:\Users\Bartek\Desktop\SUAV\whole_nomad`  
**Total Size:** 280.33 GB  
**Total Files:** 43,425 files  
**Total Directories:** 613 directories  

---

## 1. High-Level Summary & Folder Hierarchy

The `whole_nomad` folder contains a human detection/re-identification multi-altitude aerial dataset organized into 10 actor batch folders, along with 2 thumbnail/keyframe aggregate folders.

```
C:\Users\Bartek\Desktop\SUAV\whole_nomad\
├── Actor001-Actor010/          # Batch folder for Actors 1 to 10 (4,575 files, ~38.14 GB)
├── Actor011-Actor020/          # Batch folder for Actors 11 to 20 (4,250 files, ~32.38 GB)
├── Actor021-Actor030/          # Batch folder for Actors 21 to 30 (4,250 files, ~29.54 GB)
├── Actor031-Actor040/          # Batch folder for Actors 31 to 40 (4,250 files, ~26.25 GB)
├── Actor041-Actor050/          # Batch folder for Actors 41 to 50 (4,250 files, ~25.16 GB)
├── Actor051-Actor060/          # Batch folder for Actors 51 to 60 (4,250 files, ~21.24 GB)
├── Actor061-Actor070/          # Batch folder for Actors 61 to 70 (4,250 files, ~26.40 GB)
├── Actor071-Actor080/          # Batch folder for Actors 71 to 80 (4,250 files, ~27.34 GB)
├── Actor081-Actor090/          # Batch folder for Actors 81 to 90 (4,250 files, ~24.42 GB)
├── Actor091-Actor100/          # Batch folder for Actors 91 to 100 (4,250 files, ~25.61 GB)
├── TN/                         # All Keyframes / Thumbnails (500 files, ~3.21 GB)
└── TN_a50/                     # Altitude 50m Keyframes / Thumbnails (100 files, ~0.64 GB)
```

---

## 2. Detailed Internal Organization

### 2.1 Actor Batch Folders (`ActorXXX-ActorYYY`)
Each batch directory contains 10 actor subdirectories (e.g. `Actor001` through `Actor010`).

Inside each **Actor** folder (e.g., `Actor001`), there are **5 altitude subdirectories** corresponding to flight capture altitudes:
- `ActorXXX_a10/` (10m altitude)
- `ActorXXX_a30/` (30m altitude)
- `ActorXXX_a50/` (50m altitude)
- `ActorXXX_a70/` (70m altitude)
- `ActorXXX_a90/` (90m altitude)

#### Naming & Frame Format:
- **Frame files:** `ActorXXX_aYY_fZZZZ.jpg` (e.g., `Actor001_a10_f0001.jpg`)
- **Image counts:**
  - **Actor001:** 150 frame images per altitude (750 total frames across 5 altitudes)
  - **Actor002 to Actor100:** 85 frame images per altitude (425 total frames per actor across 5 altitudes)
  - **Total Actor Frames:** 42,825 JPG images

---

### 2.2 Keyframe / Thumbnail Folders

1. **`TN/`**
   - **Contents:** 500 JPG files (1 keyframe thumbnail for each actor and altitude combination).
   - **Naming Convention:** `ActorXXX_aYY_TN.jpg` (e.g., `Actor001_a10_TN.jpg`, `Actor001_a30_TN.jpg`, etc.)
   - **Size:** ~3.21 GB

2. **`TN_a50/`**
   - **Contents:** 100 JPG files (1 keyframe thumbnail for each actor specifically at 50m altitude).
   - **Naming Convention:** `ActorXXX_a50_TN.jpg` (e.g., `Actor001_a50_TN.jpg`, `Actor002_a50_TN.jpg`, etc.)
   - **Size:** ~0.64 GB

---

## 3. Directory Breakdown & File Statistics

| Directory Name | Actor Range | Folder Count | Image Count | Disk Usage |
| :--- | :--- | :---: | :---: | :---: |
| `Actor001-Actor010` | Actor001 – Actor010 | 10 Actors / 50 Subfolders | 4,575 files | 38.14 GB |
| `Actor011-Actor020` | Actor011 – Actor020 | 10 Actors / 50 Subfolders | 4,250 files | 32.38 GB |
| `Actor021-Actor030` | Actor021 – Actor030 | 10 Actors / 50 Subfolders | 4,250 files | 29.54 GB |
| `Actor031-Actor040` | Actor031 – Actor040 | 10 Actors / 50 Subfolders | 4,250 files | 26.25 GB |
| `Actor041-Actor050` | Actor041 – Actor050 | 10 Actors / 50 Subfolders | 4,250 files | 25.16 GB |
| `Actor051-Actor060` | Actor051 – Actor060 | 10 Actors / 50 Subfolders | 4,250 files | 21.24 GB |
| `Actor061-Actor070` | Actor061 – Actor070 | 10 Actors / 50 Subfolders | 4,250 files | 26.40 GB |
| `Actor071-Actor080` | Actor071 – Actor080 | 10 Actors / 50 Subfolders | 4,250 files | 27.34 GB |
| `Actor081-Actor090` | Actor081 – Actor090 | 10 Actors / 50 Subfolders | 4,250 files | 24.42 GB |
| `Actor091-Actor100` | Actor091 – Actor100 | 10 Actors / 50 Subfolders | 4,250 files | 25.61 GB |
| `TN` | All (Actors 001–100, All Altitudes) | 1 Folder | 500 files | 3.21 GB |
| `TN_a50` | All (Actors 001–100, 50m Altitude) | 1 Folder | 100 files | 0.64 GB |
| **Total** | **100 Actors** | **613 Folders** | **43,425 files** | **280.33 GB** |

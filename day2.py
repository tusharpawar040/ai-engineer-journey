import os

folder = 'test_files'
files = os.listdir(folder)
print('all files ', files)

year = '2026_'
for filename in files:
    old_path = os.path.join(folder, filename)
    new_filename = filename.lower()
    new_filename = year + new_filename
    new_path = os.path.join(folder, new_filename)
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_filename}")

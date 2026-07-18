import os
import glob
import time

print("Searching for images...")
patterns = [
    "C:/Users/User/.gemini/**/*.jpg",
    "C:/Users/User/.gemini/**/*.png",
    "C:/Users/User/.gemini/**/*.webp",
    "C:/Users/User/Desktop/**/*.jpg",
    "C:/Users/User/Desktop/**/*.png",
    "C:/Users/User/AppData/Local/Temp/**/*.jpg",
    "C:/Users/User/AppData/Local/Temp/**/*.png",
]

for p in patterns:
    for f in glob.glob(p, recursive=True):
        print(f)

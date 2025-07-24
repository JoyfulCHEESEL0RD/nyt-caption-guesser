from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import requests
import re


"""
# Set base URL
base_url = "https://www.nytimes.com"

# ✅ Step 1: Load local HTML file
file_path = "/home/nvidia/jetson-inference/python/training/classification/data/final_project/nyt_picture_column.html"

if not os.path.isfile(file_path):
    print(f"File not found or not a file: {file_path}")
    exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

# ✅ Step 2: Use regex to find correct NYT links
pattern = re.compile(
    r"^https://www\.nytimes\.com/\d{4}/\d{2}/\d{2}/learning/whats-going-on-in-this-picture-.*\.html"
)

links = []

for a in soup.find_all("a", href=True):
    href = a["href"]
    if pattern.match(href):
        links.append(href)

# Remove duplicates
unique_links = list(set(links))

# ✅ Output
if not unique_links:
    print("No matching NYT links found.")
else:
    print(f"Found {len(unique_links)} article links:\n")
    for link in unique_links:
        print(link)
"""

#################################################################


from bs4 import BeautifulSoup
import os
import json

folder_path = "articles"  # where your .html files are
data = []

for filename in sorted(os.listdir(folder_path)):
    if not filename.endswith(".html"):
        continue

    filepath = os.path.join(folder_path, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # ✅ Get caption
    caption = "No caption found"
    paragraphs = soup.find_all("p")
    for i, p in enumerate(paragraphs):
        if "The original caption reads" in p.get_text():
            if i + 1 < len(paragraphs):
                caption = paragraphs[i + 1].get_text(strip=True)
            break

    # ✅ Get image link
    # --- Image Extraction ---
    img_url = None

    meta_img = soup.find("meta", attrs={"property": "twitter:image"})
    if meta_img and meta_img.has_attr("content"):
        img_url = meta_img["content"].split("?")[0]

    # Optional fallback if you want to keep <source> / <img> as backups
    if not img_url:
        for tag in soup.find_all("source"):
            srcset = tag.get("srcset", "")
            for part in srcset.split(","):
                url_part = part.strip().split(" ")[0]
                if "static01.nyt.com/images" in url_part:
                    img_url = url_part.split("?")[0]
                    break
            if img_url:
                break

    if not img_url:
        for img in soup.find_all("img"):
            for attr in ["src", "data-src"]:
                url = img.get(attr, "")
                if "static01.nyt.com/images" in url:
                    img_url = url.split("?")[0]
                    break
            if img_url:
                break

    data.append({
        "filename": filename,
        "image_url": img_url,
        "caption": caption
    })

# ✅ Save to JSON
with open("nyt_picture_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(data)} entries into nyt_picture_data.json")



"""
############################
#EXTRACTING IMAGE LINKS 

folder_path = "articles"
for filename in sorted(os.listdir(folder_path)):
    if not filename.endswith(".html"):
        continue

    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

            
    # --- Image Extraction ---
    img_url = None

    meta_img = soup.find("meta", attrs={"property": "twitter:image"})
    if meta_img and meta_img.has_attr("content"):
        img_url = meta_img["content"].split("?")[0]

    # Optional fallback if you want to keep <source> / <img> as backups
    if not img_url:
        for tag in soup.find_all("source"):
            srcset = tag.get("srcset", "")
            for part in srcset.split(","):
                url_part = part.strip().split(" ")[0]
                if "static01.nyt.com/images" in url_part:
                    img_url = url_part.split("?")[0]
                    break
            if img_url:
                break

    if not img_url:
        for img in soup.find_all("img"):
            for attr in ["src", "data-src"]:
                url = img.get(attr, "")
                if "static01.nyt.com/images" in url:
                    img_url = url.split("?")[0]
                    break
            if img_url:
                break
    print(f"Image URL for {filename}: {img_url if img_url else 'Not found'}")

    
    # Try <source> first
    for tag in soup.find_all("source"):
        if "srcset" in tag.attrs:
            srcset = tag["srcset"]
            for part in srcset.split(","):
                url_part = part.strip().split(" ")[0]
                if "static01.nyt.com/images" in url_part:
                    img_url = url_part
                    break
        if img_url:
            break

    # Fallback: Try <img>
    if not img_url:
        for img in soup.find_all("img"):
            for attr in ["src", "data-src"]:
                url = img.get(attr, "")
                if "static01.nyt.com/images" in url:
                    img_url = url
                    break
            if img_url:
                break

    # Optional: strip query params
    if img_url and "?" in img_url:
        img_url = img_url.split("?")[0]

    """
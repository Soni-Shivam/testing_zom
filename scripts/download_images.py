import re
import os
import urllib.request
import hashlib

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(base_dir, "frontend")
templates_dir = os.path.join(frontend_dir, "templates")
static_img_dir = os.path.join(frontend_dir, "static", "images")

os.makedirs(static_img_dir, exist_ok=True)

# Process all raw HTML files
files = ["menu_raw.html", "cart_raw.html", "home_raw.html"]

for f in files:
    file_path = os.path.join(templates_dir, f)
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Simple regex to find img src tags
    # <img ... src="https://lh3.googleusercontent.com/..."
    img_urls = re.findall(r'src="(https://lh3\.googleusercontent\.com/[^"]+)"', content)
    
    print(f"File {f}: Found {len(img_urls)} images.")
    
    for url in img_urls:
        # Extract the ID from the end of the URL and hash it to prevent name length issues
        parts = url.split('/')
        img_id = parts[-1] 
        hashed_name = hashlib.md5(img_id.encode('utf-8')).hexdigest()
        local_filename = f"{hashed_name}.jpg"
        local_path = os.path.join(static_img_dir, local_filename)
        
        # Download if not exists
        if not os.path.exists(local_path):
            try:
                print(f"Downloading {local_filename}...")
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                
        # Replace the URL in the content with the local path
        local_url = f"/static/images/{local_filename}"
        content = content.replace(url, local_url)
        
    # Write back the processed HTML to a new file
    output_filename = f.replace("_raw", "")
    output_path = os.path.join(frontend_dir, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(content)
        
print("Image download and mapping complete!")

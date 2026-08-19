#!/usr/bin/env python3
"""Fix broken alt text in picture elements"""

import re
from pathlib import Path

def fix_alt_text(filepath):
    """Fix alt="png" to proper alt text"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Find and fix pattern where alt text got corrupted
    # Look for patterns like: alt="png" that should be alt="Description"
    # We'll extract from the original img src filename

    # Pattern: find <picture> blocks with alt="png"
    pattern = r'<picture>\s*<source srcset="([^"]+\.webp)" type="image/webp">\s*<img\s+src="([^"]+\.(png|jpg))"\s+alt="(png|jpg)"\s+loading="lazy"\s+decoding="async"\s*/>\s*</picture>'

    def replacement(match):
        webp_src = match.group(1)
        img_src = match.group(2)

        # Generate alt text from filename
        filename = Path(img_src).stem
        alt_text = filename.replace('-', ' ').replace('_', ' ').title()

        return f'''<picture>
  <source srcset="{webp_src}" type="image/webp">
  <img
    src="{img_src}"
    alt="{alt_text}"
    loading="lazy"
    decoding="async"
  />
</picture>'''

    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed alt text in {filepath.name}")
        return True
    return False

# Fix the file
filepath = Path('/Users/louisesfer/Documents/Programming/portfolio/project-cinescope.html')
if fix_alt_text(filepath):
    print("Alt text has been corrected!")
else:
    print("No changes needed or pattern not found")

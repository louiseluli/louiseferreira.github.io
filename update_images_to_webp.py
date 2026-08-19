#!/usr/bin/env python3
"""
Update HTML files to use WebP images with PNG fallbacks and lazy loading
"""

import re
import os
from pathlib import Path

def convert_img_to_picture(match):
    """Convert <img> tag to <picture> with WebP support and lazy loading"""
    indent = match.group(1)
    src = match.group(2)
    alt = match.group(3)

    # Get WebP path
    webp_src = src.replace('.png', '.webp').replace('.jpg', '.webp')

    # Build picture element
    picture = f'''{indent}<picture>
{indent}  <source srcset="{webp_src}" type="image/webp">
{indent}  <img
{indent}    src="{src}"
{indent}    alt="{alt}"
{indent}    loading="lazy"
{indent}    decoding="async"
{indent}  />
{indent}</picture>'''

    return picture

def update_html_file(filepath):
    """Update a single HTML file"""
    print(f"\nProcessing: {filepath.name}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern to match <img> tags with src and alt
    # Captures indentation, src path, and alt text
    pattern = r'(\s+)<img\s+src="([^"]+\.(png|jpg))"\s+alt="([^"]+)"\s*/?>'

    # Count matches
    matches = re.findall(pattern, content)
    print(f"  Found {len(matches)} images to convert")

    # Replace all img tags
    content = re.sub(pattern, convert_img_to_picture, content)

    # Write back if changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Updated {len(matches)} images")
        return len(matches)
    else:
        print(f"  No changes needed")
        return 0

def main():
    """Main execution"""
    portfolio_dir = Path('/Users/louisesfer/Documents/Programming/portfolio')

    print("=" * 60)
    print("HTML Image Optimization: WebP + Lazy Loading")
    print("=" * 60)

    # Files to update
    html_files = [
        'project-cinescope.html',
        'project-algofairness.html',
        'project-cinema.html',
        'project-docscope.html',
        'index.html',
        'about.html',
        'experience.html',
        'projects.html',
        'skills.html',
        'awards.html',
        'personal.html',
        'contact.html',
    ]

    total_updated = 0

    for filename in html_files:
        filepath = portfolio_dir / filename
        if filepath.exists():
            count = update_html_file(filepath)
            total_updated += count
        else:
            print(f"\n⚠ File not found: {filename}")

    print("\n" + "=" * 60)
    print(f"✓ Complete! Updated {total_updated} images across {len(html_files)} files")
    print("=" * 60)
    print("\nChanges:")
    print("  • All images now use <picture> elements")
    print("  • WebP format served to modern browsers")
    print("  • PNG/JPG fallback for older browsers")
    print("  • loading=\"lazy\" added for better performance")
    print("  • decoding=\"async\" for smoother rendering")
    print()

if __name__ == '__main__':
    main()

#!/bin/bash

# CineScope Portfolio Image Optimization and Addition Script
# Simplified version with direct copy approach

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== CineScope Image Optimization ===${NC}"
echo ""

QUALITY=85
CINESCOPE_SOURCE="/Users/louisesfer/Documents/Programming/CineScope/analysis_outputs/visualizations"
PORTFOLIO_CINESCOPE="assets/img/cinescope"

# Navigate to portfolio
cd /Users/louisesfer/Documents/Programming/portfolio

# Function to convert and report
convert_and_report() {
    local png="$1"
    local webp="${png%.png}.webp"

    if [ -f "$webp" ]; then
        return
    fi

    echo -e "${GREEN}Converting: $(basename "$png")${NC}"
    cwebp -q $QUALITY "$png" -o "$webp" 2>/dev/null

    local png_size=$(du -h "$png" | cut -f1)
    local webp_size=$(du -h "$webp" | cut -f1)
    echo "  $png_size → $webp_size"
}

echo -e "${BLUE}Step 1: Converting existing CineScope images to WebP...${NC}"
cd "$PORTFOLIO_CINESCOPE"

existing_count=0
for png in *.png; do
    if [ -f "$png" ]; then
        convert_and_report "$png"
        ((existing_count++))
    fi
done

echo ""
echo -e "${GREEN}✓ Converted $existing_count existing images${NC}"
echo ""

# Copy new visualizations
echo -e "${BLUE}Step 2: Adding new visualizations...${NC}"
echo ""

# Define images to copy (source:destination pairs)
new_images=(
    # Network Analysis - impressive visual storytelling
    "batch_31/network_overview.png:network-overview.png"
    "batch_31/centrality_analysis.png:centrality-analysis.png"
    "batch_31/frequent_costars.png:frequent-costars.png"

    # Release Patterns
    "batch_20/seasonal_performance.png:seasonal-performance.png"
    "batch_20/release_density_calendar.png:release-calendar-heatmap.png"

    # Completeness Analysis
    "batch_34/completeness_overview.png:actor-completeness-overview.png"

    # Financial Analysis
    "batch_24/genre_box_office_performance.png:genre-box-office.png"
    "batch_24/revenue_distribution.png:revenue-distribution-detailed.png"

    # Narrative Analysis
    "batch_25/plot_themes.png:plot-themes.png"
    "batch_25/narrative_structure.png:narrative-structure.png"

    # Awards
    "batch_29/awards_overview.png:awards-comprehensive.png"
    "batch_29/oscar_deep_dive.png:oscar-deep-dive.png"
)

added_count=0
for mapping in "${new_images[@]}"; do
    source_path="${mapping%%:*}"
    dest_name="${mapping##*:}"
    full_source="$CINESCOPE_SOURCE/$source_path"

    if [ ! -f "$full_source" ]; then
        echo -e "${YELLOW}⚠ Not found: $source_path${NC}"
        continue
    fi

    if [ -f "$dest_name" ]; then
        echo -e "${YELLOW}Already exists: $dest_name${NC}"
        convert_and_report "$dest_name"
        continue
    fi

    echo -e "${BLUE}Adding: $dest_name${NC}"
    cp "$full_source" "$dest_name"
    convert_and_report "$dest_name"
    ((added_count++))
    echo ""
done

echo ""
echo -e "${GREEN}✓ Added $added_count new images${NC}"
echo ""

# Calculate savings
echo -e "${BLUE}Step 3: Calculating size savings...${NC}"
echo ""

png_total=$(du -ch *.png 2>/dev/null | grep total | cut -f1)
webp_total=$(du -ch *.webp 2>/dev/null | grep total | cut -f1)
png_count=$(ls -1 *.png 2>/dev/null | wc -l | tr -d ' ')
webp_count=$(ls -1 *.webp 2>/dev/null | wc -l | tr -d ' ')

echo "CineScope Image Inventory:"
echo "  PNG files:  $png_count ($png_total)"
echo "  WebP files: $webp_count ($webp_total)"
echo ""

png_bytes=$(du -sk *.png 2>/dev/null | awk '{sum+=$1} END {print sum}')
webp_bytes=$(du -sk *.webp 2>/dev/null | awk '{sum+=$1} END {print sum}')

if [ -n "$png_bytes" ] && [ -n "$webp_bytes" ] && [ "$png_bytes" -gt 0 ]; then
    savings=$((100 - (webp_bytes * 100 / png_bytes)))
    echo -e "${GREEN}Size reduction: ~${savings}%${NC}"
    echo "  Original: ${png_total}"
    echo "  Optimized: ${webp_total}"
fi

cd ../..

echo ""
echo -e "${GREEN}=== Optimization Complete! ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Update project-cinescope.html with new images"
echo "  2. Add <picture> elements for WebP support"
echo "  3. Add loading=\"lazy\" to all images"
echo ""

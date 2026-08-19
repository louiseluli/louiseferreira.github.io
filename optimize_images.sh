#!/bin/bash

# CineScope Portfolio Image Optimization Script
# Converts PNG images to WebP format with quality optimization
# Maintains original PNGs as fallbacks

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CineScope Image Optimization Script ===${NC}"
echo ""

# Configuration
QUALITY=85  # WebP quality (0-100, 85 is good balance)
PORTFOLIO_IMG_DIR="assets/img"
CINESCOPE_SOURCE="/Users/louisesfer/Documents/Programming/CineScope/analysis_outputs/visualizations"

# Function to convert PNG to WebP
convert_to_webp() {
    local png_file="$1"
    local webp_file="${png_file%.png}.webp"

    if [ -f "$webp_file" ]; then
        echo -e "${YELLOW}  Skipping (WebP exists): $(basename "$webp_file")${NC}"
        return
    fi

    echo -e "${GREEN}  Converting: $(basename "$png_file")${NC}"
    cwebp -q $QUALITY "$png_file" -o "$webp_file" 2>/dev/null

    # Show file size comparison
    local png_size=$(du -h "$png_file" | cut -f1)
    local webp_size=$(du -h "$webp_file" | cut -f1)
    echo -e "    PNG: $png_size → WebP: $webp_size"
}

# Function to optimize existing portfolio images
optimize_existing() {
    echo -e "${BLUE}Step 1: Optimizing existing portfolio images...${NC}"
    echo ""

    cd "$PORTFOLIO_IMG_DIR"

    # Optimize main images
    echo "Main images:"
    for img in vincent-price.jpg venom-mob.jpg; do
        if [ -f "$img" ]; then
            webp_file="${img%.jpg}.webp"
            if [ ! -f "$webp_file" ]; then
                echo -e "${GREEN}  Converting: $img${NC}"
                cwebp -q $QUALITY "$img" -o "$webp_file" 2>/dev/null
                du -h "$img" "$webp_file" | awk '{print "   ", $2, $1}'
            fi
        fi
    done
    echo ""

    # Optimize CineScope images
    echo "CineScope images:"
    cd cinescope
    local count=0
    local total=$(ls -1 *.png 2>/dev/null | wc -l | tr -d ' ')

    for png in *.png; do
        if [ -f "$png" ]; then
            ((count++))
            echo -e "${BLUE}[$count/$total]${NC}"
            convert_to_webp "$png"
        fi
    done

    cd ../..
    echo ""
    echo -e "${GREEN}✓ Existing images optimized${NC}"
    echo ""
}

# Function to copy and optimize new CineScope visualizations
copy_new_visualizations() {
    echo -e "${BLUE}Step 2: Adding new CineScope visualizations...${NC}"
    echo ""

    # Selected visualizations to add (curated for portfolio impact)
    declare -A new_images=(
        # Network Analysis (Batch 31) - Compelling visual impact
        ["batch_31/network_overview.png"]="network-overview.png"
        ["batch_31/centrality_analysis.png"]="centrality-analysis.png"
        ["batch_31/frequent_costars.png"]="frequent-costars.png"

        # Release Timing (Batch 20) - Data storytelling
        ["batch_20/seasonal_performance.png"]="seasonal-performance.png"
        ["batch_20/release_density_calendar.png"]="release-calendar-heatmap.png"

        # Actor Completeness (Batch 34) - Personal insight
        ["batch_34/completeness_overview.png"]="actor-completeness-overview.png"

        # Box Office (Batch 24) - Commercial patterns
        ["batch_24/genre_box_office_performance.png"]="genre-box-office.png"
        ["batch_24/revenue_distribution.png"]="revenue-distribution.png"

        # Plot/Story Analysis (Batch 25) - Narrative patterns
        ["batch_25/plot_themes.png"]="plot-themes.png"
        ["batch_25/narrative_structure.png"]="narrative-structure.png"

        # Awards Deep Dive (Batch 29) - Prestige tracking
        ["batch_29/awards_overview.png"]="awards-comprehensive.png"
        ["batch_29/oscar_deep_dive.png"]="oscar-deep-dive.png"
    )

    cd "$PORTFOLIO_IMG_DIR/cinescope"

    local count=0
    local total=${#new_images[@]}

    for source_path in "${!new_images[@]}"; do
        ((count++))
        local dest_name="${new_images[$source_path]}"
        local full_source="$CINESCOPE_SOURCE/$source_path"

        echo -e "${BLUE}[$count/$total]${NC} Adding: $dest_name"

        if [ ! -f "$full_source" ]; then
            echo -e "${YELLOW}  ⚠ Source not found: $source_path${NC}"
            continue
        fi

        # Copy PNG
        cp "$full_source" "$dest_name"
        echo -e "${GREEN}  ✓ Copied PNG${NC}"

        # Convert to WebP
        convert_to_webp "$dest_name"
    done

    cd ../..
    echo ""
    echo -e "${GREEN}✓ New visualizations added${NC}"
    echo ""
}

# Function to calculate total size savings
calculate_savings() {
    echo -e "${BLUE}Step 3: Calculating size savings...${NC}"
    echo ""

    cd "$PORTFOLIO_IMG_DIR/cinescope"

    local png_total=$(du -ch *.png 2>/dev/null | grep total | cut -f1)
    local webp_total=$(du -ch *.webp 2>/dev/null | grep total | cut -f1)
    local png_count=$(ls -1 *.png 2>/dev/null | wc -l | tr -d ' ')
    local webp_count=$(ls -1 *.webp 2>/dev/null | wc -l | tr -d ' ')

    echo "Results:"
    echo "  PNG files:  $png_count ($png_total total)"
    echo "  WebP files: $webp_count ($webp_total total)"
    echo ""

    # Calculate percentage savings
    local png_bytes=$(du -ck *.png 2>/dev/null | grep total | awk '{print $1}')
    local webp_bytes=$(du -ck *.webp 2>/dev/null | grep total | awk '{print $1}')

    if [ -n "$png_bytes" ] && [ -n "$webp_bytes" ] && [ "$png_bytes" -gt 0 ]; then
        local savings=$((100 - (webp_bytes * 100 / png_bytes)))
        echo -e "${GREEN}  Size reduction: ~${savings}%${NC}"
    fi

    cd ../..
    echo ""
}

# Main execution
main() {
    # Navigate to portfolio root
    cd /Users/louisesfer/Documents/Programming/portfolio

    # Run optimization steps
    optimize_existing
    copy_new_visualizations
    calculate_savings

    echo -e "${GREEN}=== Optimization Complete! ===${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Update HTML files to use <picture> elements with WebP"
    echo "  2. Add loading=\"lazy\" to all <img> tags"
    echo "  3. Test portfolio in browser"
    echo ""
}

# Run main function
main

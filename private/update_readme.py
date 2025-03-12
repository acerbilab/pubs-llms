#!/usr/bin/env python3
# AI Summary: Automatically updates README.md with a formatted list of publications extracted from markdown files.
# Extracts BibTeX data, checks for companion files, and sorts publications by year and venue.
# Script runs from the ./private folder but operates on files in the root directory.

import os
import re
import glob
from pathlib import Path
from collections import defaultdict

# Set up the repository root path (one level up from the private folder)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def extract_bibtex(file_path):
    """Extract BibTeX information from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for BibTeX content between triple backticks
    bibtex_match = re.search(r'```\s*@\w+{([^,]+),\s*title={([^}]*)},\s*author={([^}]*)},\s*journal={([^}]*)},\s*year={(\d+)}', content, re.DOTALL)
    
    if not bibtex_match:
        print(f"Warning: Could not extract BibTeX from {file_path}")
        return None
    
    return {
        'key': bibtex_match.group(1),
        'title': bibtex_match.group(2),
        'authors': bibtex_match.group(3),
        'venue': bibtex_match.group(4),
        'year': int(bibtex_match.group(5)),
        'file_path': file_path
    }

def format_author_list(authors):
    """Format the author list for display as 'Surname FI' with '& ' for last author."""
    author_list = [author.strip() for author in authors.split('and')]
    formatted_authors = []
    
    for author in author_list:
        parts = author.split(',')
        if len(parts) > 1:
            # Lastname, Firstname format
            lastname = parts[0].strip()
            firstname = parts[1].strip()
            # Get initials from firstname
            initials = ''.join([name[0] for name in firstname.split() if name])
            formatted_authors.append(f"{lastname} {initials}")
        else:
            # Firstname Lastname format
            name_parts = author.split()
            if len(name_parts) > 1:
                lastname = name_parts[-1]
                # Get initials from firstname parts
                initials = ''.join([name[0] for name in name_parts[:-1] if name])
                formatted_authors.append(f"{lastname} {initials}")
            else:
                formatted_authors.append(author)
    
    # Join with commas but use '& ' for the last author unless there's only one author
    if len(formatted_authors) == 1:
        return formatted_authors[0]
    elif len(formatted_authors) > 1:
        return ', '.join(formatted_authors[:-1]) + ' & ' + formatted_authors[-1]
    else:
        return ''

def get_conference_order(venue):
    """Return a number to sort conferences within a year."""
    venue_lower = venue.lower()
    # Approximate conference dates (month) for sorting
    if 'neurips' in venue_lower or 'nips' in venue_lower:
        return 12  # December
    elif 'icml' in venue_lower:
        return 7   # July
    elif 'iclr' in venue_lower:
        return 5   # May
    elif 'aistats' in venue_lower:
        return 4   # April
    elif 'uai' in venue_lower:
        return 8   # August
    elif 'ijcai' in venue_lower:
        return 8   # August
    elif 'aaai' in venue_lower:
        return 2   # February
    else:
        # Default to bottom for unknown venues
        return 0

def update_readme():
    """Update the README.md file with the list of publications."""
    # Get frontmatter content from private/frontmatter.md
    frontmatter_path = os.path.join(REPO_ROOT, 'private', 'frontmatter.md')
    try:
        with open(frontmatter_path, 'r', encoding='utf-8') as f:
            frontmatter = f.read()
    except FileNotFoundError:
        print(f"Warning: {frontmatter_path} not found, using empty frontmatter")
        frontmatter = ""
    
    # Find all main markdown files (exclude _appendix and _backmatter)
    main_files = []
    for md_file in glob.glob(os.path.join(REPO_ROOT, '*.md')):
        filename = os.path.basename(md_file)
        if '_appendix' not in filename and '_backmatter' not in filename and filename != 'README.md':
            main_files.append(md_file)
    
    # Extract BibTeX information from each file
    publications = []
    for file_path in main_files:
        bibtex_data = extract_bibtex(file_path)
        if bibtex_data:
            # Check for companion files
            base_name = Path(file_path).stem
            appendix_file = os.path.join(REPO_ROOT, f"{base_name}_appendix.md")
            backmatter_file = os.path.join(REPO_ROOT, f"{base_name}_backmatter.md")
            
            bibtex_data['has_appendix'] = os.path.exists(appendix_file)
            bibtex_data['has_backmatter'] = os.path.exists(backmatter_file)
            bibtex_data['appendix_path'] = appendix_file if bibtex_data['has_appendix'] else None
            bibtex_data['backmatter_path'] = backmatter_file if bibtex_data['has_backmatter'] else None
            
            # Format GitHub links
            file_basename = os.path.basename(file_path)
            bibtex_data['github_link'] = f"https://github.com/acerbilab/pubs-llms/blob/main/{file_basename}"
            
            if bibtex_data['has_appendix']:
                appendix_basename = os.path.basename(appendix_file)
                bibtex_data['appendix_github_link'] = f"https://github.com/acerbilab/pubs-llms/blob/main/{appendix_basename}"
            
            if bibtex_data['has_backmatter']:
                backmatter_basename = os.path.basename(backmatter_file)
                bibtex_data['backmatter_github_link'] = f"https://github.com/acerbilab/pubs-llms/blob/main/{backmatter_basename}"
            
            publications.append(bibtex_data)
    
    # Sort publications by year (descending) and then by conference order
    publications.sort(key=lambda x: (-x['year'], -get_conference_order(x['venue'])))
    
    # Group publications by year
    publications_by_year = defaultdict(list)
    for pub in publications:
        publications_by_year[pub['year']].append(pub)
    
    # Generate formatted list of publications
    publications_md = "## Publications\n\n"
    
    for year in sorted(publications_by_year.keys(), reverse=True):
        publications_md += f"### {year}\n\n"
        
        for pub in publications_by_year[year]:
            # Format authors
            formatted_authors = format_author_list(pub['authors'])
            
            # Create publication entry
            publications_md += f"- **{pub['title']}**<br>\n"
            publications_md += f"  {formatted_authors}<br>\n"
            publications_md += f"  *{pub['venue']}*<br>\n"
            
            # Add simplified navigation links
            nav_links = []
            nav_links.append(f"[main]({pub['github_link']})")
            if pub['has_appendix']:
                nav_links.append(f"[appendix]({pub['appendix_github_link']})")
            if pub['has_backmatter']:
                nav_links.append(f"[backmatter]({pub['backmatter_github_link']})")
            
            publications_md += f"  {' | '.join(nav_links)}\n"
            
            publications_md += "\n"
    
    # Combine frontmatter and publications list
    readme_content = f"{frontmatter}\n\n{publications_md}"
    
    # Write to README.md
    readme_path = os.path.join(REPO_ROOT, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README.md updated with {len(publications)} publications")

if __name__ == "__main__":
    update_readme()

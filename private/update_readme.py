#!/usr/bin/env python3
# AI Summary: Automatically updates README.md with a formatted list of publications extracted from markdown files.
# Extracts BibTeX data, checks for companion files, and sorts publications by year and venue.
# Uses a two-step venue mapping process with special handling for workshop venues.

import os
import re
import glob
from pathlib import Path
from collections import defaultdict

# Set up the repository root path (one level up from the private folder)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Venue standardization mapping - maps variant spellings to standard names
VENUE_STANDARDIZATION = {
    # NeurIPS variants
    'Advances in Neural Information Processing Systems': 'Neural Information Processing Systems',
    'Proceedings of Neural Information Processing Systems': 'Neural Information Processing Systems',
    'NeurIPS': 'Neural Information Processing Systems',
    'NIPS': 'Neural Information Processing Systems',
    # ICML variants
    'Proceedings of the International Conference on Machine Learning': 'International Conference on Machine Learning',
    'ICML': 'International Conference on Machine Learning',
    # ICLR variants
    'Proceedings of the International Conference on Learning Representations': 'International Conference on Learning Representations',
    'ICLR': 'International Conference on Learning Representations',
    # AISTATS variants
    'Proceedings of Artificial Intelligence and Statistics': 'Artificial Intelligence and Statistics',
    'AISTATS': 'Artificial Intelligence and Statistics',
    # UAI variants
    'Proceedings of the Conference on Uncertainty in Artificial Intelligence': 'Conference on Uncertainty in Artificial Intelligence',
    'UAI': 'Conference on Uncertainty in Artificial Intelligence',
    # IJCAI variants
    'Proceedings of the International Joint Conference on Artificial Intelligence': 'International Joint Conference on Artificial Intelligence',
    'IJCAI': 'International Joint Conference on Artificial Intelligence',
    # AAAI variants
    'Proceedings of the AAAI Conference on Artificial Intelligence': 'AAAI Conference on Artificial Intelligence',
    'AAAI': 'AAAI Conference on Artificial Intelligence',
    # Journal variants
    'PLOS Computational Biology': 'PLoS Computational Biology',
    'PLoS Comput Biol': 'PLoS Computational Biology',
    'PLOS ONE': 'PLoS ONE',
    'Comput Brain Behav': 'Computational Brain & Behavior',
    'Journal of Open Source Software': 'Journal of Open Source Software',
    'JOSS': 'Journal of Open Source Software',
}

# Venue abbreviation mapping - maps standard names to abbreviations
VENUE_ABBREVIATIONS = {
    # Conferences
    'Neural Information Processing Systems': 'NeurIPS',
    'International Conference on Machine Learning': 'ICML',
    'International Conference on Learning Representations': 'ICLR',
    'Artificial Intelligence and Statistics': 'AISTATS',
    'Conference on Uncertainty in Artificial Intelligence': 'UAI',
    'International Joint Conference on Artificial Intelligence': 'IJCAI',
    'AAAI Conference on Artificial Intelligence': 'AAAI',
    # Journals
    'PLoS Computational Biology': 'PLoS Comput Biol',
    'PLoS ONE': 'PLoS ONE',
    'Computational Brain & Behavior': 'Comput Brain Behav',
    'Computational Brain \& Behavior': 'Comput Brain Behav',
    'Journal of Open Source Software': 'JOSS',
}

def identify_workshop(venue):
    """Check if the venue is a workshop."""
    return 'workshop' in venue.lower()

def standardize_venue_name(venue):
    """Standardize venue name by mapping variant spellings to standard names."""
    # Check for exact matches first
    if venue in VENUE_STANDARDIZATION:
        return VENUE_STANDARDIZATION[venue]
    
    # Check for substring matches if no exact match
    for variant, standard in VENUE_STANDARDIZATION.items():
        if variant.lower() in venue.lower():
            return standard
    
    # Return original if no match found
    return venue

def get_venue_abbreviation(venue):
    """Get abbreviation for a standardized venue name, with workshop handling."""
    # First standardize the venue name
    standard_venue = standardize_venue_name(venue)
    
    # Check if this is a workshop
    is_workshop = identify_workshop(venue)
    
    # Get abbreviation of the standardized name
    if standard_venue in VENUE_ABBREVIATIONS:
        abbr = VENUE_ABBREVIATIONS[standard_venue]
    else:
        abbr = standard_venue
    
    # Append "Workshop" if it's a workshop
    if is_workshop and " Workshop" not in abbr:
        abbr += " Workshop"
    
    return abbr

def extract_bibtex(file_path):
    """Extract BibTeX information from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for BibTeX content between triple backticks
    bibtex_match = re.search(r'```\s*@\w+{([^,]+),\s*title={([^}]*)},\s*author={([^}]*)},\s*(?:journal|booktitle)={([^}]*)},\s*year={(\d+)}', content, re.DOTALL)
    
    if not bibtex_match:
        # Try a more flexible approach for different BibTeX formats
        bibtex_block = re.search(r'```(.*?)```', content, re.DOTALL)
        if bibtex_block:
            bibtex_content = bibtex_block.group(1)
            
            # Extract individual fields
            key_match = re.search(r'@\w+{([^,]+),', bibtex_content)
            title_match = re.search(r'title={([^}]*)}', bibtex_content)
            author_match = re.search(r'author={([^}]*)}', bibtex_content)
            venue_match = re.search(r'(?:journal|booktitle)={([^}]*)}', bibtex_content)
            year_match = re.search(r'year={(\d+)}', bibtex_content)
            
            if key_match and title_match and author_match and venue_match and year_match:
                return {
                    'key': key_match.group(1),
                    'title': title_match.group(1),
                    'authors': author_match.group(1),
                    'venue': venue_match.group(1),
                    'year': int(year_match.group(1)),
                    'file_path': file_path
                }
        
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
    import re
    author_list = [author.strip() for author in re.split(r'\s+and\s+', authors)]
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
    
    # Initialize default value
    venue_value = 0
    
    # Approximate conference dates (month) for sorting
    if 'neurips' in venue_lower or 'nips' in venue_lower:
        venue_value = 12  # December
    elif 'icml' in venue_lower:
        venue_value = 7   # July
    elif 'iclr' in venue_lower:
        venue_value = 5   # May
    elif 'aistats' in venue_lower:
        venue_value = 4   # April
    elif 'uai' in venue_lower:
        venue_value = 8   # August
    elif 'ijcai' in venue_lower:
        venue_value = 8   # August
    elif 'aaai' in venue_lower:
        venue_value = 2   # February
    
    # Subtract 0.5 if venue is a workshop
    if 'workshop' in venue_lower or 'workshops' in venue_lower:
        venue_value -= 0.5
        
    return venue_value

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
    
    # Find all main markdown files (look for _main suffix)
    main_files = []
    for md_file in glob.glob(os.path.join(REPO_ROOT, 'publications', '*_main.md')):
        main_files.append(md_file)
    
    # Extract BibTeX information from each file
    publications = []
    for file_path in main_files:
        bibtex_data = extract_bibtex(file_path)
        if bibtex_data:
            # Check for companion files
            # Extract base name by removing "_main" suffix
            base_name = Path(file_path).stem.replace("_main", "")
            publications_dir = os.path.join(REPO_ROOT, 'publications')
            appendix_file = os.path.join(publications_dir, f"{base_name}_appendix.md")
            backmatter_file = os.path.join(publications_dir, f"{base_name}_backmatter.md")
            
            bibtex_data['has_appendix'] = os.path.exists(appendix_file)
            bibtex_data['has_backmatter'] = os.path.exists(backmatter_file)
            bibtex_data['appendix_path'] = appendix_file if bibtex_data['has_appendix'] else None
            bibtex_data['backmatter_path'] = backmatter_file if bibtex_data['has_backmatter'] else None
            
            # Format GitHub links
            file_basename = os.path.basename(file_path)
            bibtex_data['github_link'] = f"https://github.com/acerbilab/pubs-llms/blob/main/publications/{file_basename}"
            
            if bibtex_data['has_appendix']:
                appendix_basename = os.path.basename(appendix_file)
                bibtex_data['appendix_github_link'] = f"https://github.com/acerbilab/pubs-llms/blob/main/publications/{appendix_basename}"
            
            if bibtex_data['has_backmatter']:
                backmatter_basename = os.path.basename(backmatter_file)
                bibtex_data['backmatter_github_link'] = f"https://github.com/acerbilab/pubs-llms/blob/main/publications/{backmatter_basename}"
            
            # Get venue abbreviation
            bibtex_data['venue_abbr'] = get_venue_abbreviation(bibtex_data['venue'])
            
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
            
            # Add links with venue abbreviation
            nav_links = []
            nav_links.append(f"`{pub['venue_abbr']}`")
            nav_links.append(f"[main]({pub['github_link']})")
            if pub['has_appendix']:
                nav_links.append(f"[appendix]({pub['appendix_github_link']})")
            if pub['has_backmatter']:
                nav_links.append(f"[backmatter]({pub['backmatter_github_link']})")
            
            publications_md += f"  {' | '.join(nav_links)}\n\n"
    
    # Combine frontmatter and publications list
    readme_content = f"{frontmatter}\n\n{publications_md}"
    
    # Write to README.md
    readme_path = os.path.join(REPO_ROOT, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README.md updated with {len(publications)} publications")

if __name__ == "__main__":
    update_readme()

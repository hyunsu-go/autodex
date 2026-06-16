# AutoDex — Project Page

Static project page for **AutoDex: An Automated Real-World System for Dexterous Grasping Data Collection** (CoRL 2026).

## Structure
```
website/
├── index.html              # the page (single file)
└── static/
    ├── css/index.css       # styles
    ├── images/             # teaser, system overview, result figures (PNG)
    └── videos/             # teaser + pipeline + result clips (web-sized MP4)
```

## Preview locally
```bash
cd website
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy (GitHub Pages)
Push the contents of `website/` to a `gh-pages` branch (or point Pages at this folder):
```bash
git subtree push --prefix website origin gh-pages
```

## TODO before publishing
- Fill in the **Paper**, **arXiv**, and **Code** button URLs in `index.html` (currently `href="#"`).
- Update the BibTeX once the citation is final.
- Confirm author email addresses are correct.

Template adapted from the [Nerfies](https://github.com/nerfies/nerfies.github.io) project page.

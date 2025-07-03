---
layout: default
title: Search
permalink: /search/
---

# Search the Course

Use the search box above to find content throughout this speech recognition course. The search will look through all modules, lessons, and documentation for matching content.

## Search Features

- **Intelligent Search**: Powered by Lunr.js for fast, client-side full-text search
- **Keyboard Navigation**: Use ↑/↓ arrow keys to navigate results, Enter to open
- **Quick Access**: Press `/` from anywhere on the site to focus the search box
- **Highlighted Results**: Search terms are highlighted in results for easy scanning
- **Content Types**: Results show whether content is from a page or post
- **Categories & Tags**: Additional metadata helps contextualize results

## What You Can Search For

- **Technical terms** (e.g., "acoustic model", "language model", "speech recognition")
- **Module topics** (e.g., "signal processing", "decoding", "neural networks")  
- **Code functions** (e.g., "HTK", "feature extraction", "training")
- **Concepts** (e.g., "MFCC", "HMM", "CTC", "beam search")
- **File formats** (e.g., "WAV", "ARPA", "MLF")
- **Tools and frameworks** (e.g., "CNTK", "Kaldi", "OpenFST")

<div id="search-info" style="margin-top: 20px; padding: 15px; background: #f6f8fa; border-radius: 6px; border-left: 4px solid #0366d6;">
  <p><strong>💡 Pro Tip:</strong> Start typing in the search box above to find relevant course content instantly. Use specific terms for better results.</p>
</div>

## Browse by Module

Alternatively, you can browse the course content by module:

- **[Module 1: Introduction]({{ "/M1_Introduction/" | relative_url }})** - Overview and scoring metrics
- **[Module 2: Speech Signal Processing]({{ "/M2_Speech_Signal_Processing/" | relative_url }})** - Feature extraction and signal processing
- **[Module 3: Acoustic Modeling]({{ "/M3_Acoustic_Modeling/" | relative_url }})** - HMM and neural network acoustic models
- **[Module 4: Language Modeling]({{ "/M4_Language_Modeling/" | relative_url }})** - N-gram and neural language models
- **[Module 5: Decoding]({{ "/M5_Decoding/" | relative_url }})** - Search algorithms and beam search
- **[Module 6: End-to-End Models]({{ "/M6_End_to_End_Models/" | relative_url }})** - Modern neural approaches

<script>
  // Focus search input when on search page
  document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
      searchInput.focus();
    }
  });
</script>

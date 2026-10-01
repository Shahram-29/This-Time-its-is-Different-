# Study and planning documents

Every document comes as **Word (.docx, editable)** and **PDF (for reading; bookmarks in the side panel)**.
These are study and planning documents, not dissertation text.

| Open this… | …when you want to | PDF | Word |
|---|---|---|---|
| **Project Handbook** | understand or explain any part of the project: terms, methods, scripts, graphs, results, viva questions. **Start here.** v1.1 (30 Sep 2026) adds the exploratory extensions E1–E2 (E10, F12–F13, G15–G17, H9–H10). | [PDF](Project_Handbook.pdf) | [docx](Project_Handbook.docx) |
| **Research Plan** | see the design decisions, the verified facts behind them and the week-by-week plan. v1.4 (30 Sep 2026): results status (Section 12), Amendment A1 (5.14), corrections | [PDF](Dissertation_Research_Plan.pdf) | [docx](Dissertation_Research_Plan.docx) |
| **Chapter 2 Blueprint** | plan Chapter 2 section by section; hypothesis derivation table; synthesis matrix; reference list. Updated 30 Sep 2026: new sources for the exploratory extensions; Gallagher & Twomey correction | [PDF](Chapter2_Literature_Review_Blueprint.pdf) | [docx](Chapter2_Literature_Review_Blueprint.docx) |
| **Literature Handbook v2** | study the 61 core papers (cards with page pointers) and the 11 figures from your data (updated 1 Oct 2026: 10 cards and 2 figures for E3–E4) | [PDF](Literature_Handbook_Chapter2_v2.pdf) | [docx](Literature_Handbook_Chapter2_v2.docx) |
| **Literature Search Guide** | see what the 28 Sep 2026 searches found, how each paper relates to your results, and what to read first. Updated 30 Sep 2026: exploratory results added; corrections applied | [PDF](Literature_Search_Guide_Strands_1-2.pdf) | [docx](Literature_Search_Guide_Strands_1-2.docx) |
| archive / Literature Handbook v1 | superseded by v2 — kept for the record only | [PDF](archive/Literature_Handbook_Chapter2.pdf) | [docx](archive/Literature_Handbook_Chapter2.docx) |

**Suggested reading order:** Project Handbook → Research Plan → Chapter 2 Blueprint → Literature Handbook v2 → Literature Search Guide.

**After editing a Word file**, re-create the PDFs (Word must be installed; takes about a minute). Open a PowerShell terminal in the
Dissertation folder and run the command below. It works in a separate hidden copy of Word, so any document you have open stays as it is.

```
powershell -ExecutionPolicy Bypass -File docs\make_pdfs.ps1
```

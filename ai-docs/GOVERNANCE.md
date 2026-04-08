# Documentation Governance

This document defines the rules for AI contributors modifying documentation in the outomem project. These rules ensure consistency, accuracy, and alignment with project principles.

## 1. Philosophy Reference Requirement
Every document must reference `philosophy.md`. If a change contradicts a principle in `philosophy.md`, you must provide an explicit justification in the commit message. This ensures conceptual integrity across the system.

## 2. Document Types and Quality Bars
Documentation falls into four categories, each with specific requirements.

### Reference Docs
Accuracy is the priority. Every API claim must cite a `sourcefile:line` reference. Code examples must match actual function signatures exactly.

### Guides
Testability is required. Every code block must be runnable. You must include the expected output for every example.

### Explanations
Depth is expected. Go beyond surface descriptions. You must include limitations and tradeoffs for every feature or component.

### Philosophy
Conceptual integrity is the goal. Principles must map directly to code implementations.

## 3. Cross-Reference Integrity
Every internal link must resolve. Broken links are documentation bugs. Use relative paths for all internal links.

## 4. Korean-First Language Rule
Korean strings from source code, such as sentiment keywords, prompts, or fallback messages, must appear verbatim. Don't translate or paraphrase Korean content. This preserves the original intent and technical accuracy.

## 5. Source Code Grounding
Every API document must cite `file:line` references. Claims about system behavior must be verifiable in the source code. If you can't find the code that supports a claim, don't make the claim.

## 6. AI Contribution Checklist
Complete this checklist before submitting documentation changes:
- [ ] All internal links resolve to existing files.
- [ ] All code examples match actual function signatures.
- [ ] All file:line citations are accurate.
- [ ] Korean content matches source code exactly.
- [ ] Changes don't contradict philosophy.md principles.
- [ ] Cross-references are updated if you added new sections.

## 7. Deprecation Policy
When code changes, mark affected documentation with a `[STALE]` header note. Don't delete the content. Mark it as stale with the date and the reason for the change.

## 8. Architecture Decisions
When you document design decisions, link to the relevant `philosophy.md` principle. Don't document "what" without explaining "why".

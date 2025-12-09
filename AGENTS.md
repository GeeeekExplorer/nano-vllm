## Tool Priority

- Filename search: `fd`.
- Text/content search: `rg` (ripgrep).
- AST/structural search: `sg` (ast-grep) — preferred for code-aware queries (imports, call expressions, JSX/TSX nodes).

### AST-grep Usage (Linux)

- Announce intent and show the exact command before running complex patterns.
- Common queries:
  - Find imports from `node:path` (TypeScript/TSX):
    - `ast-grep -p "import $$ from 'node:path'" src --lang ts,tsx,mts,cts`
  - Find CommonJS requires of `node:path`:
    - `ast-grep -p "require('node:path')" src --lang js,cjs,mjs,ts,tsx`
  - Suggest rewrite (do not auto-apply in code unless approved):
    - Search: `ast-grep -p "import $$ from 'node:path'" src --lang ts,tsx`
    - Proposed replacement: `import $$ from 'pathe'`

### Search Hygiene (fd/rg/sg)

- Exclude bulky folders to keep searches fast and relevant: `.git`, `node_modules`, `coverage`, `out`, `dist`.
- Prefer running searches against a scoped path (e.g., `src`) to implicitly avoid vendor and VCS directories.
- Examples:
  - `rg -n "pattern" -g "!{.git,node_modules,coverage,out,dist}" src`
  - `fd --hidden --exclude .git --exclude node_modules --exclude coverage --exclude out --exclude dist --type f ".tsx?$" src`
- ast-grep typically respects `.gitignore`; target `src` to avoid scanning vendor folders:
  - `ast-grep -p "import $$ from '@shared/$$'" src --lang ts,tsx,mts,cts`
  - If needed, add ignore patterns to your ignore files rather than disabling ignores.
# Create a standalone repository

Use these commands when this branch is ready and you want a clean repository
without the experiment history from the source project.

## 1. Commit the framework state in this repository

```powershell
git status --short
git add -A
git commit -m "Create general radiomics framework"
```

## 2. Export only the current tree to a new folder

```powershell
cd C:\Users\jalzate\Documents
New-Item -ItemType Directory -Force General-Radiomics-Framework
$sourceRepo = "C:\Users\jalzate\Documents\<current-source-folder>"
git -C $sourceRepo archive --format=tar HEAD |
  tar -x -C General-Radiomics-Framework
```

## 3. Initialize the new independent repository

```powershell
cd C:\Users\jalzate\Documents\General-Radiomics-Framework
git init
git add -A
git commit -m "Initial general radiomics framework"
```

## 4. Connect it to a new remote

Create an empty repository on GitHub, GitLab, Azure DevOps, or another Git
server, then run:

```powershell
git branch -M main
git remote add origin https://github.com/<user-or-org>/general-radiomics-framework.git
git push -u origin main
```

Replace the remote URL with the real one.

## Alternative: keep history

If you want to preserve the full Git history, clone only this branch instead.
This is not recommended if the old repository contains large experiment files
or sensitive data in previous commits.

```powershell
cd C:\Users\jalzate\Documents
$sourceRepo = "C:\Users\jalzate\Documents\<current-source-folder>"
git clone --single-branch --branch codex-general-radiomics-framework $sourceRepo General-Radiomics-Framework
cd General-Radiomics-Framework
git remote remove origin
git remote add origin https://github.com/<user-or-org>/general-radiomics-framework.git
git push -u origin codex-general-radiomics-framework:main
```

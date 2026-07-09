#!/usr/bin/env bash
# 모노레포 → lhnet-inpainting-lab 조직 컴포넌트 레포 동기화.
#
# 사용법 (레포 안 어디서든, Git Bash):
#   bash scripts/sync-org-repos.sh          # dev 브랜치 기준
#   bash scripts/sync-org-repos.sh main     # main 브랜치 기준
#
# 동작: 디렉터리별 커밋 이력을 subtree split으로 다시 뽑아 조직 레포 main에
# 미러 푸시한다(force). 조직 레포는 미러이므로 거기에 직접 커밋하지 말 것 —
# 직접 커밋하면 다음 동기화 때 사라진다. README도 모노레포 쪽에서 관리한다.
#
# lhnet-research(demo·docs·experiments)는 자주 안 바뀌어 스크립트에서 제외.
# 필요할 때 수동 동기화:
#   git clone --no-local . /tmp/research && cd /tmp/research
#   python -m git_filter_repo --path demo --path docs --path experiments --force
#   git push --force https://github.com/lhnet-inpainting-lab/lhnet-research.git HEAD:main
set -euo pipefail

BRANCH="${1:-dev}"
ORG="lhnet-inpainting-lab"

cd "$(git rev-parse --show-toplevel)"

if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  echo "브랜치 없음: $BRANCH" >&2
  exit 1
fi

for pair in backend:lhnet-backend frontend:lhnet-frontend model:lhnet-model; do
  dir="${pair%%:*}"
  repo="${pair##*:}"
  echo "== $dir → $ORG/$repo"
  sha="$(git subtree split -P "$dir" "$BRANCH")"
  git push --force "https://github.com/$ORG/$repo.git" "$sha:refs/heads/main"
done

echo "동기화 완료 ($BRANCH 기준)"

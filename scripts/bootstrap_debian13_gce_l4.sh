#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
PYTHON_VERSION="${PYTHON_VERSION:-}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev,gpu,generation,vision}"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

log() {
  printf '[bootstrap] %s\n' "$*"
}

ensure_components_line() {
  local file_path="$1"
  local tmp_path

  tmp_path="$(mktemp)"

  awk '
    /^Components:/ {
      split("main contrib non-free non-free-firmware", ordered, " ")
      out = "Components:"
      for (i = 1; i <= length(ordered); i++) {
        out = out " " ordered[i]
      }
      print out
      next
    }
    { print }
  ' "${file_path}" > "${tmp_path}"

  if ! cmp -s "${file_path}" "${tmp_path}"; then
    ${SUDO} cp "${file_path}" "${file_path}.bak"
    ${SUDO} cp "${tmp_path}" "${file_path}"
    log "Updated ${file_path} to include contrib/non-free/non-free-firmware."
  fi

  rm -f "${tmp_path}"
}

enable_debian_repos() {
  if [[ -f /etc/apt/sources.list.d/debian.sources ]]; then
    ensure_components_line "/etc/apt/sources.list.d/debian.sources"
    return
  fi

  if [[ -f /etc/apt/sources.list ]]; then
    local tmp_path
    tmp_path="$(mktemp)"

    awk '
      /^deb / {
        if ($0 !~ / contrib([[:space:]]|$)/) {
          $0 = $0 " contrib"
        }
        if ($0 !~ / non-free([[:space:]]|$)/) {
          $0 = $0 " non-free"
        }
        if ($0 !~ / non-free-firmware([[:space:]]|$)/) {
          $0 = $0 " non-free-firmware"
        }
      }
      { print }
    ' /etc/apt/sources.list > "${tmp_path}"

    if ! cmp -s /etc/apt/sources.list "${tmp_path}"; then
      ${SUDO} cp /etc/apt/sources.list /etc/apt/sources.list.bak
      ${SUDO} cp "${tmp_path}" /etc/apt/sources.list
      log "Updated /etc/apt/sources.list to include contrib/non-free/non-free-firmware."
    fi

    rm -f "${tmp_path}"
    return
  fi

  log "No Debian apt sources file found. Update repositories manually before continuing."
  exit 1
}

install_apt_packages() {
  log "Refreshing apt metadata."
  ${SUDO} apt-get update

  log "Installing build and driver prerequisites."
  ${SUDO} apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    dkms \
    git \
    libbz2-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    llvm \
    make \
    patch \
    pciutils \
    pkg-config \
    tk-dev \
    uuid-dev \
    wget \
    xz-utils \
    zlib1g-dev

  if ! ${SUDO} apt-get install -y --no-install-recommends "linux-headers-$(uname -r)"; then
    log "Kernel-specific headers unavailable; falling back to linux-headers-amd64."
    ${SUDO} apt-get install -y --no-install-recommends linux-headers-amd64
  fi

  log "Installing NVIDIA driver packages."
  ${SUDO} apt-get install -y --no-install-recommends \
    firmware-misc-nonfree \
    nvidia-driver
}

setup_pyenv() {
  if [[ ! -d "${PYENV_ROOT}" ]]; then
    log "Installing pyenv into ${PYENV_ROOT}."
    git clone https://github.com/pyenv/pyenv.git "${PYENV_ROOT}"
  fi

  export PYENV_ROOT
  export PATH="${PYENV_ROOT}/bin:${PATH}"

  if ! command -v pyenv >/dev/null 2>&1; then
    log "pyenv is not available on PATH after installation."
    exit 1
  fi
}

resolve_python_bin() {
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return
  fi

  setup_pyenv

  if [[ -z "${PYTHON_VERSION}" ]]; then
    PYTHON_VERSION="$(pyenv install --list | sed 's/^[[:space:]]*//' | awk '/^3\.12\.[0-9]+$/ { print }' | tail -n 1)"
  fi

  if [[ -z "${PYTHON_VERSION}" ]]; then
    log "Unable to resolve a Python 3.12 version through pyenv."
    exit 1
  fi

  log "Installing Python ${PYTHON_VERSION} with pyenv."
  pyenv install -s "${PYTHON_VERSION}"
  printf '%s/bin/python\n' "${PYENV_ROOT}/versions/${PYTHON_VERSION}"
}

create_or_update_venv() {
  local python_bin="$1"

  log "Creating virtual environment at ${VENV_DIR}."
  "${python_bin}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  log "Upgrading pip tooling."
  python -m pip install --upgrade pip setuptools wheel

  log "Installing project dependencies."
  if [[ -n "${INSTALL_EXTRAS}" ]]; then
    python -m pip install -e ".[${INSTALL_EXTRAS}]"
  else
    python -m pip install -e .
  fi
}

main() {
  if [[ ! -f /etc/debian_version ]]; then
    log "This script expects Debian."
    exit 1
  fi

  enable_debian_repos
  install_apt_packages

  local python_bin
  python_bin="$(resolve_python_bin)"
  create_or_update_venv "${python_bin}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi; then
      log "nvidia-smi is installed but the driver is not ready yet. A reboot is usually required."
    fi
  else
    log "nvidia-smi is not available yet. Check the NVIDIA package install and reboot if required."
  fi

  log "Bootstrap complete."
  log "Next step: ${PROJECT_ROOT}/scripts/verify_environment.sh"
}

main "$@"

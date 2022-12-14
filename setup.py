# fmt: off
import os
import os.path as op

from setuptools import setup


DISTNAME = "ephysvibe"
DESCRIPTION = "."
MAINTAINER = "Camila Losada"
MAINTAINER_EMAIL = "camilaalosada@gmail.com"
URL = "https://github.com/camilosada/EphysVibe"
LICENSE = ""
DOWNLOAD_URL = "https://github.com/camilosada/EphysVibe"
VERSION = ""


def parse_requirements_file(fname):
    requirements = list()
    with open(fname, 'r') as fid:
        for line in fid:
            req = line.strip()
            if req.startswith('#'):
                continue
            # strip end-of-line comments
            req = req.split('#', maxsplit=1)[0].strip()
            requirements.append(req)
    return requirements


def package_tree(pkgroot):
    """Get the submodule list."""
    path = op.dirname(__file__)
    subdirs = [
        op.relpath(i[0], path).replace(op.sep, '.')
        for i in os.walk(op.join(path, pkgroot))
        if '__init__.py' in i[2]
    ]
    return sorted(subdirs)

def get_version(distname):
    version = None
    # get the version
    # (don't import ephysvibe here, so dependencies are not needed)
    with open(op.join(distname, '_version.py'), 'r') as fid:
        for line in (line.strip() for line in fid):
            if line.startswith("__version__"):
                version = line.split('=')[1].strip().strip('\'')
                break
    if version is None:
        raise RuntimeError("Could not determine the version.")

    return version


if __name__ == "__main__":

    if op.exists('MANIFEST'):
        os.remove('MANIFEST')

    try:
        VERSION = get_version(DISTNAME)
    except Exception as ex:
        raise ex

    with open('README.md', 'r') as fid:
        long_description = fid.read()

    install_requires = parse_requirements_file('requirements.txt')
   
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=long_description,
        long_description_content_type="text/markdown",
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Human Machine Interfaces",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
        ],
        keywords="electrophysiology neuroscience",
        project_urls={
            "Source": "https://github.com/camilosada/EphysVibe",
            "Tracker": "https://github.com/camilosada/EphysVibe/issues/",
        },
        platforms="any",
        python_requires=">=3.7",
        install_requires=install_requires,
        packages=package_tree("ephysvibe"),
    )

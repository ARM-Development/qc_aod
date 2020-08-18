#!/bin/sh

proctype='vap'
process='qc_aod'
package="$proctype-$process"
execname=$process

python_prefix='/apps/base/python3.6'
python="$python_prefix/bin/python"
pip="$python_prefix/bin/pip"

# Print usage

script_name=$0

usage()
{
    cat <<EOM

DESCRIPTION

  Build script used to create a local install of this package. It can also
  be used to preform a staged installation by specifying a destdir, in which
  case the files are copied instead of linked. 

SYNOPSIS

  $script_name [--prefix=path] [--destdir=path] [--conf] [--package]

OPTIONS

  --prefix=path   absolute path to installation directory for conf files
                  and link to the executable
                  default: \$VAP_HOME

  --destdir=path  absolute path prepended to prefix used
                  to perform a staged installation

  --conf          only install the conf files

  --package       only install the python package

  --uninstall     uninstall locally installed package

  -h, --help      display this help message

EOM
}

# Parse command line

for i in "$@"
do
    case $i in
        --prefix=*)       prefix="${i#*=}"
                          ;;
        --destdir=*)      destdir="${i#*=}"
                          ;;
        --conf)           conf_only=1
                          ;;
        --package)        pkg_only=1
                          ;;
        --uninstall)      uninstall=1
                          ;;
        -h | --help)      usage
                          exit 0
                          ;;
        *)                usage
                          exit 1
                          ;;
    esac
done

# Get prefix and install_data from environemnt variables if necessary

if [ ! $prefix ]; then
    if [ $VAP_HOME ]; then
        prefix=$VAP_HOME
    else
        usage
        exit 1
    fi
fi

# Functions to echo and run commands

run() {
    echo "> $1"
    $1 || exit 1
}

# Install conf files

if [ ! $pkg_only ] && [ -d conf ]; then

    echo "------------------------------------------------------------------"
    confdir=$destdir$prefix/conf/$proctype
    if [ ! -d $confdir ]; then
        run "mkdir -p $confdir"
    fi

    confdir+=/$process
    run "rm -f $confdir"

    if [ ! $uninstall ]; then
        if [ $destdir ]; then
            run "mkdir -p $confdir"
            run "cp -R conf/* $confdir"
        else 
            pwd=`pwd`
            run "ln -s $pwd/conf $confdir"
        fi
    fi
fi

# Install Python Package

if [ ! $conf_only ]; then

    echo "------------------------------------------------------------------"
    bindir=$destdir$prefix/bin
    if [ ! -d $bindir ]; then
        run "mkdir -p $bindir"
    fi

    run "rm -f $bindir/$execname"

    if [ $uninstall ]; then
        if [ $destdir ]; then
            echo "Cannot uninstall package from root path: $destdir"
            echo "The $package package must be uninstalled manually."
        else
            run "rm -f $HOME/.local/bin/$execname"
            run "$pip uninstall $package"
        fi
    else
        if [ $destdir ]; then
            run "$python setup.py install --root=$destdir"
            run "ln -s ../../..$python_prefix/bin/$execname $bindir/$execname"
        else
            run "$pip install --user -e ."
            run "ln -s $HOME/.local/bin/$execname $bindir/$execname"
        fi
    fi
fi

exit 0


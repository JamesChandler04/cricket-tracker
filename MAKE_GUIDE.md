# Make Guide

## Installing Make on Windows

Run the following command or go to [here](https://gnuwin32.sourceforge.net/packages/make.htm) and follow the download instructions.

```
winget install ezwinports.make
```

## Installing Make on MacOS

You can check if you have Homebrew installed by running this command:

```
brew -v
```

If the version comes up, you have it installed, if an error is raised, you need to run the following command:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then using Homebrew, you can download Make using the following command:

```
brew install make
```

## Using the Makefile

This project includes a `Makefile` to make it easier to run. Here are the list of commands:

- `make setup`: Create virtual environment.
- `make install`: Install required package in virtual environment (only have to do this once per venv).
- `make run`: Runs the program.
- `make clean`: Remove virtual environment.

To setup a new environment, run `make setup` and `make install`.
While you still have the [.venv](./.venv) folder, you can jsut run `make run` to run the program.
You only have to rerun `make install` if you update [requirements.txt](./requirements.txt).
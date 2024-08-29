# Contributing to nrt

Thanks for taking the time to contribute to nrt! 🎉

## Rights

The EUPL v2 license (see LICENSE) applies to all contributions.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue and include as much detail as possible. Include steps to reproduce the issue and any relevant logs or screenshots.

### Suggesting Enhancements

Enhancement suggestions are welcome! If you have an idea to improve nrt or its documentation, please open an issue and describe your idea in detail. If possible, provide examples of how the enhancement would be used.

### Code Contributions

For any contribution to the code base or the documentation, use the pull request mechanism.
1. Fork the repository: Click the 'Fork' button on the upper right corner of the repository page.
2. Apply changes to your fork.
3. Open a pull request on github


Your contribution will be reviewed and discussied as part of the pull request. If approved, it will then be merged
into the main branch of the repository and included in the following release. 


### Testing

We use `pytest` for unit tests.

- Unit tests are written using the `pytest` framework.
- Tests are automatically run using GitHub CI with every push and pull request.
- You can run tests locally by simply calling `pytest` in the root directory of the project.


### Releasing a new version

Package version is set via git tags thanks to [setuptools-scm](https://setuptools-scm.readthedocs.io/en/latest/). A new release
is made for every tagged commit pushed to github and that passes unit tests.
Examples git tag command: `git tag -a v0.3.0 -m "version 0.3.0"


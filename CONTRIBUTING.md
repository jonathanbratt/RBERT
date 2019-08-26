## Contributing

Want to contribute? Great! First, read this page (including the small print at the end).

### Before you contribute
Before we can use your code, you must sign the [Macmillan Learning Individual Contributor License Agreement](https://docs.google.com/document/d/1DSw306V4eFzi4UJQbXk1MVFZ0QNzxa4AMrHnQ16UpyI) (CLA). The CLA is necessary mainly because you own the copyright to your changes, even after your contribution becomes part of our codebase, so we need your permission to use and distribute your code. We also need to be sure of various other things—for instance that you'll tell us if you know that your code infringes on other people's patents. You don't have to sign the CLA until after you've submitted your code for review and a member has approved it, but you must do it before we can put your code into our codebase. 
Before you start working on a larger contribution, you should get in touch with us first through the issue tracker with your idea so that we can help out and possibly guide you. Coordinating up front makes it much easier to avoid frustration later on.

### Making changes

We use the github [fork and pull review process](https://help.github.com/articles/using-pull-requests) to review all contributions. First, fork the repository by following the [github instructions](https://help.github.com/articles/fork-a-repo). Then check out your personal fork:

    $ git clone https://github.com/<username>/RBERT.git

Add an upstream remote so you can easily keep up to date with the main repository:

    $ git remote add upstream https://github.com/macmillanlearning-open/RBERT.git

To update your local repo from the main:

    $ git pull upstream master

When you're done making changes, make sure tests pass, and then commit your changes to your personal fork. Then use the GitHub Web UI to create and send the pull request. We'll review and merge the change.


### Code review

All submissions, including submissions by project members, require review. To keep the code base maintainable and readable, all code is developed using a similar coding style. We typically follow the [tidyverse style guide](https://style.tidyverse.org/), with minor changes.


The code should be easy to maintain and understand. It is important that you be able to come back, months later, to code that you've written and still quickly understand what it is supposed to be doing. Understandable code also makes it easier for other people to contribute. Quick-and-dirty solutions or "clever" coding tricks might work in the short term, but should be avoided in the interest of long term code quality.

With the code review process, we ensure that at least two sets of eyes looked over the code in hopes of finding potential bugs or errors (before they become bugs and errors). This also improves the overall code quality and makes sure that every developer knows to (largely) expect the same coding style.


[Unit tests](https://testthat.r-lib.org/) are an important part of the code. We aim for 100% test coverage, while recognizing that some functionality may be hard to cover in a unit test. 



### The small print 

Contributions made by corporations will be covered by a
different agreement than the one above. Contact us if this applies to you.
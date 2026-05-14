# Nvidia ECT Coding Conventions

## Formatting and Styling

Our conventions for style started with the [Linux kernel coding
style](https://www.kernel.org/doc/html/v4.10/process/coding-style.html), and
have evolved slightly from there.  In terms of editor / auto formatting tools,
the Linux kernel styles that are almost certainly built in are a good place to
start.

### Tabs

We use tabs with a width of 8.

### Breaking long lines and strings

We no longer access our computers via [VT100
consoles](https://en.wikipedia.org/wiki/VT100) (just emulated ones), but we do
sometimes work on small laptops in coffee shops.  With 8 space tabs and good
variable names, 80 characters is a bit restrictive so we extend the Linux kernel
guidance of 80 characters to 100 characters.  But if the line is close, consider
splitting the line to help those working on little laptops.

### Placing Braces and Spaces

The use of unbraced conditional bodies is strictly forbidden.  Too many bugs
have been caused by the “harmless debugging printf” that changed whether a
statement was part of the conditional body or not.

### Typedefs

We generally disagree with Linus on this point, but especially so given that the
plugin is written in C++.  The use of `_t` to indicate a type is consistent in
our code base due to the history of being C code, but should not be used when
creating new types.

### Centralized exiting of functions

Our code base started with following this paradigm, and when things must be
unwrapped before exit, it’s a good starting point.  Gotos are bad, except when
you’re unwinding your function.  In code that has been updated to be proper C++,
gotos should be avoided if RAII-style code can be used instead.  For example,
instead of writing:

```
pthread_mutex_t data_lock;

int manipulate_data(void)
{
    int ret;
    pthread_mutex_lock(&data_lock);

    ret = do_thing_1();
    if (ret != 0) {
        goto unlock;
    }

    ret = do_thing_2();
    if (ret != 0) {
        goto unlock;
    }

    ret = 0;

 unlock:
    pthread_mutex_unlock(&data_lock);
    return ret;
 }
```

you should use `std::lock_guard`:

```
std::mutex data_lock;

int manipulate_data(void)
{
    int ret;
    std::lock_guard l(lock_guard);

    ret = do_thing_1();
    if (ret != 0) return ret;

    ret = do_think_2();
    if (ret != 0) return ret;

    return 0;
}
```

Some of the libraries we use are not going to follow an RAII pattern and this
may make things more complicated.  For example, LTTNG traces on entrance and
exit may be best expressed with goto-style unwinding.  Use your judgement, and
don’t over-complexify to be pure about exit patterns.

### “struct” prefix in front of struct types

For things that are C structs (like Libfabric objects), use the full struct
fid_cq. For internal things, drop the type specifier.

## Conventions

### Exceptions

In C++ catching and throwing an exception is crazy expensive, but having
exception handling code active is cheap.  This leads to a some obvious
conventions:

* If the operation that failed is retriable (for example, a submission queue
  being full or a completion queue being empty), return an error code rather
  than throw an exception.  A practical outcome of this is that most functions
  will have C-style int return codes.
* Allocation failures are rare, and handling them properly is near impossible.
  There’s also no real recovery in this case for the plugin (not like we can
  free other people’s memory).  Throw an exception rather than build complicated
  handling.  Just make sure to clean up your resources on the way out (RAII is
  your friend).
* Use your judgement on the rest.  If the error is just going to propagate to
  NCCL as a system error, then throwing an exception is probably the right call.
  If something between the error and NCCL is likely to take the error, handle
  it, and move on, a return code is probably the right call.

This is a place where judgement is necessary.  Unless its in the absolute
performance critical part of the code (like handling completion queues), bias
towards what makes the code most readable and maintainable.

### Naming

Naming is hard and the source of much [bike
shedding](https://en.wikipedia.org/wiki/Law_of_triviality), but we will take a
stab at some conventions anyway.

* Variable names should be both descriptive and short.  Don’t do anything
  perverse by removing random characters or all vowels, but also `tmp` is not a
  good variable name if it is used multiple times across a 100 line function.
* For functions not part of a class, functions should generally be named
  `<Noun>_<verb>_<object>`, for example `rdma_domain_get_device().`  With C++
  this is likely to become `rdma_domain.get_device()`.  Functions which do not
  make it clear which objects they operate on are likely to cause confusion.
  For example, `init_``rails()` requires more thought than
  `rdma_domain_init_rails()` for the next developer.

### Combine allocation and initialization

Combine allocation and initialization of resources into a single place. For
example:

```
rdma_domain_t *rdma_create_domain(...)
{
    ...
    domain->rails = rdma_create_domain_rails(..., num_rails, ...);
    ...
 }
```

is significantly easier to understand or modify than code that splits the two,
such as:

```
rdma_domain_t *rdma_create_domain(...)
{
    ...
    domain->rails = calloc(num_rails, sizeof(rdma_domain_rail_t));
    ...
    rdma_domain_rails_init(domain->rails);
    ...
 }
```

### Using Inline

When adding new class member functions or refactoring existing stand-alone
functions to be members: if it's a short function, put the function definition
in the class declaration with the `inline` keyword. If it's a long function, put
it's declaration in the separate source file without the `inline` keyword.


## Commits and Pull Requests

Each Pull Request should implement one feature.  For example, adding support for
a new NCCL API version or a significant stage in a new feature.  Each git commit
in a Pull Request should compile and work independent of follow-on commits, so
that the tree is always properly bisectable.  Each commit should also be a
simple, standalone change.  This is, of course, a judgement call, but the idea
is to have each commit be something that reviewers can get their head around,
and reviewers are much more likely to catch the important details in your
commits if each one is a single, documented logical change.

### Pull Request Descriptions

A Pull Request subject and body description should describe the entire pull
request, and does not need to go into details about each commit.  For single
commit PRs, using the Git commit subject and body (which GitHub will do
automatically for you) is generally sufficient.

### Git Commit Messages

The Git repository is the official history of our project.  GitHub pull request
data may come and go, tickets, sims, and the like will be deprecated at some
point, but Git history will live on.  Therefore, Git History is an important
development artifact for future developers.  There are many (similar) screeds on
the internet about writing proper git messages.  The author’s
[favorite](https://cbea.ms/git-commit/) is a good mix of advice and snark that
gets the point across well.  It can be summarized as 7 rules:

1. Separate subject from body with a blank line
2. Limit the subject line to 50 characters
3. Capitalize the subject line
4. Do not end the subject line with a period
5. Use the imperative voice in the subject line
6. Wrap the body at 72 characters
7. Use the body to explain BOTH what and why (vs how, unless the how helps with what or why)

The project has generally (but not always) followed the convention that commit
message subjects start with `<code area>:`.  For example, “`rdma: Remove
implicit local MR support`” immediately tells the reader scrolling through the
git history looking for changes to the send/recv protocol that they can move on.
Sometimes, the change doesn’t fit neatly into a code area and the prefix is
omitted.  The project does not recommend using Conventional Commit prefixes of
the type of commit (bugfix/feature/etc.) because 1) 50 characters is not that
many and 2) it leads to unnecessary discussions about whether a change is a
feature, a code cleanup, or a bugfix.  Just document what the change does and
everyone will be happy(-ish).

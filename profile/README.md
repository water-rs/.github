# Water Project
Water Project is an initiative aimed at benefiting the ecosystem of Rust GUI. It is comprising two main crates:

- `waterui` : The crate for building UI.
- `waterkit` : Providing native capabilities for Rust, it can easily be transplanted to other platforms.

Water Project hopes that writing GUI apps in Rust should be fun and easy, facilitating code reuse across platforms.

## The philosophy of Water

### General code works for general case, Specific code works for a specific case

"Write once, run anywhere" is unrealistic. If you force to construction abstraction to cover all the case,
it would be more complex than writing each specific case individually!

### Respect the features of every platform

Every platform has its own features, the interactive model of a PC and a mobile phone is **totally different** .
Respect these features and design for each platform accordingly.

Even among mobile phones, each has its own features! Android follows the Material Design and iOS follows HIG;
therefore, your app should truly mimic a native experience.

### Never premature optimization

“Premature optimization is the root of all evil” is a famous saying among software developers. And it is true.
You should only optimize for performance bottleneck. And ignore the tiny thing.

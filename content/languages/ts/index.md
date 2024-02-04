---
title: Typescript
---
`**Note: this section is under construction**`


Welcome to my notes on `Typescript`. I first became interested in Typescript after listening to a friend at a dinner party talk about how much better his life became after Google migrated his project to Typescript from Javscript. By then, I'd had a good handle on the various programming paradigms and *conceptually* why a strongly and statically typed language could be useful, but hadn't really felt it for myself.

Since then, I've begun using Typescript in most of my projects- but I wanted to get a better grasp on the patterns that were most fitting of this langauge.

And [Effective Typescript by Dan Vanderkam](http://google.com) fit the bill perfectly. It goes beyond mechanics to cover a set of rules and concepts that you should deeply comprehened to proficiently use Typescript in real-world use cases.


## the why and what

Before we dive into `Typescript`, let's understand why you should care about it in the first place.

`Typescript` is a strongly typed language which builds on Javascript. It supports static typing, type inference, multile platform/browsers, and goes a long way to prevent common programming mistakes / runtime errors. It's maintained by Microsoft.


It's useful for 
- Enhancing code quality and readability through static typing.
- Catching errors at compile-time, reducing runtime errors.
- Facilitating large-scale application development with better structuring and maintainability.


To give you a quick look into the difference between JavaScript and Typescript, here are two short code samples side by side.

Javascript $\Downarrow$
```javascript
function addNumbers(a, b) {
    return a + b;
}

let result = addNumbers(5, '10'); // JavaScript doesn't catch type mismatch here
console.log(result); // Outputs: '510', as '10' is treated as a string
```

Typescript $\Downarrow$
```typescript
function addNumbers(a: number, b: number): number {
    return a + b;
}

let result = addNumbers(5, '10'); // TypeScript will throw an error here
console.log(result);
```

## things you can do with this

If you're anything like me, you'd love to learn this for no reason- but it's good to know what skills you can expect to learn with this content.

With the material in these pages you should be able to 
- Critcize any piece of Typescript code you see on the internet for not adhering to proper design principles.
- Migrate your organization from Javascript to Typescript effectively and unlock the benefits it provides.

## the content

I'd recommend reading in order of the files, but I've tried to make the information as atomic as possible- enjoy!

1. [[/1|Section 1]] useful for this that and the other
2. [[/1|Section 1]] useful for this that and the other
3. [[/1|Section 1]] useful for this that and the other
4. [[/1|Section 1]] useful for this that and the other


Sources
1. Gitbook- https://gibbok.github.io/typescript-book/book/differences-between-type-and-interface/
2. Effective Typescript book https://effectivetypescript.com/


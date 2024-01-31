---
title: Built-in APIs
---

TODO

# createContext
Lets you create a context which components can provide or read
- More notes in the managing state section

# ForwardRef
`const SomeComponent = forwardRef(render)`
Expose DOM node ot parent component with a `ref`
- More notes in the escape hatches section

# lazy
Defer loading component’s code until rendered for the first time
`const SomeComponent = lazy(load)`
You can also lazy-oad React component
`const MarkdownPreview = lazy(() => import('./MarkdownPreview.js'));`

Parameters
- `load` returns a promise
Returns
- React component which you can render in your tree

More details [here](https://react.dev/reference/react/lazy)

# memo
Skip re-rendering componetn when its props are unchanged
`const MemoizedComponent = memo(SomeComponent, arePropsEqual?)`
Further notes [here](https://react.dev/reference/react/memo)
TODO

# startTransition
Update the state without blocking the UI
`startTransition(scope)`
Further notes [here](https://react.dev/reference/react/startTransition)
TODO

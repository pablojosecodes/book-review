---
title: Built-in APIs
---
This section is the only one that is incomplete- please refer to [React’s API documentation](https://react.dev/reference/react/apis) for more details
# createContext
Lets you create a context which components can provide or read
- More notes in the [[Managing State]] section
# ForwardRef
`const SomeComponent = forwardRef(render)`
Expose DOM node to parent component with a `ref`
- More notes in the [[Refs]] section

# lazy
Defer loading component’s code until rendered for the first time
`const SomeComponent = lazy(load)`

You can also lazy load React components
`const MarkdownPreview = lazy(() => import('./MarkdownPreview.js'));`

Parameters
- `load` returns a promise
Returns
- React component which you can render in your tree

More details [here](https://react.dev/reference/react/lazy)

# memo
Skip re-rendering component when its props are unchanged
`const MemoizedComponent = memo(SomeComponent, arePropsEqual?)`
Further notes [here](https://react.dev/reference/react/memo)


# startTransition
Update the state without blocking the UI
`startTransition(scope)`
Further notes [here](https://react.dev/reference/react/startTransition)


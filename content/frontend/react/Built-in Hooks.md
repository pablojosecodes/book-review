---
title: Built-in Hooks
---

# State
`useState`
- `const [index, setIndex] = useState(0);`
- Use for: state variable which you can update directly
`useReducer`
- `const [state, dispatch] = useReducer(reducer, initialArg, init?)
- Use for: state variable with update logic in reducer function
`
# Context
`useContext`: reads and subscribes to a context
- `const them = useContext(ThemeContext);`

# Ref
`useRef`: declares a ref (usually hold a DOM node)
- `const inputRef = useRef(null);.... <Component ref={inputRef}>`

# Effect
Remember- “escape hatch”

`useEffect`: connect to external system
Rarely used variations
- `useLayouEffect`: fires before browser repaints the screen
- `useInsertionEffects`: fires before React changes DOM

# Performance
Skip unnecessary work
`useMemo`: cache an expensive claculation
- `const visibleTodos = useMemo(() => filterTodos(todos, tab), [todos, tab]);`
`useCallback`: cache function definition before passing to an optimized component
- `const cachedFn = useCallback(fn, dependencies)`

To prioritize rendering
`useTransition` marks a state transition as non-blocking
- `const [isPending, startTransition] = useTransition()`
`useDeferredValue`to defer updating non-critical part of the UI
- `const deferredValue = useDeferredValue(value)`


# Resources
Accessing resource without having them be part of their state
`use` to read value of resource like `Promise` or `context`
- `const message = use(messagePromise);`
- `const theme = use(ThemeContext);`


# Other
mostly for library authors

`useDebugValue` cusomize the label React DevTools displays for your custom Hook
- `useDebugValue(value, format?)`
`useId`  associate a unique ID with (component’s self) itself. Typically used with accessibility APIs
- `const id = useId()`
`useSyncExternalStore` lets a component subscribe to an external store
- `const snapshot = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot?)`

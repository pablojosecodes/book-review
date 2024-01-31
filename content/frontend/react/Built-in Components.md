---
title: Built-in Components
---

# `< Fragment >` or `<>...</>`
Purpose: group elements without a wrapper ndoe
Example $\Downarrow$ 
```jsx
<> /* or <Fragment > */
	<OneChild />
	<AnotherChild />
</> /* or </Fragment> */
```
Props
- optional `key` 

# `<Profiler>`
Measure the rendering performance of a React tree
Sample usage $\Downarrow$ 
```jsx
<Profiler id="App" onRender={onRender}>
  <App />
</Profiler>
```
Wrap a component tree with it to measur rendering performance
Props
- `id`: identify part of the UI you’re measuring
- `onRender`: callback which React calls each time the profile tree updates
	- `function onRender(id, phase, actualDuration, baseDuration, startTime, commitTime) {...}`
Note: 
- Disabled in production build by default
- Can be nested / used for different sections

The callback itself 	`function onRender(id, phase, actualDuration, baseDuration, startTime, commitTime) {...}`
- `id` prop of Profiler ree that has just commited
- More parameters in the [documentation](https://react.dev/reference/react/Profiler) itself
	- Mostly just measure the durations of the rendering
# `<Suspense>`
Display a fallback until its children have finished loading
Sample usage $\Downarrow$ 
```jsx
<Suspense fallback={<Loading />}>
  <SomeComponent />
</Suspense>
```
Only Suspense-enabled data sources will activate the component
- Data fetching 
- Lazy loading component code with `lazy` 
- Reading the value of a promise  with `use`
Does not detect when data is fetched in effect or event handler
Further notes in [the reference material](https://react.dev/reference/react/Suspense)
- You can reveal nesteed content as it loads
- You can show stale content as it loads

# `<StrictMode>`
Enables extra development-only behavior to find warnings
Usage
```jsx
<StrictMode>
	<App />
</StrictMode>
```
Specific behaviors
- Components re-render exra time
- Re-run effects an exra time
- Check for usage of depracated APIs

You can use for whole app or just part
More details in the [docs](https://react.dev/reference/react/StrictMode)
---
title: Refs
---
# Remember info without re-rendering
Use `ref` to remember information without triggering new renders (ie. `const ref = useRef(0)`)
- Refs are retained between renders, but don’t trigger re-render
- Access current value through `ref.current`
- Are mutable
- Don’t read/write `current` value during rendering

When to use?
- When component needs to step outside React and comunicate with external APIs
	- TimeoutIDs
	- Manipualting DOM elements
	- Other objects which aren’t necessary to calculate JSX
- When storing value which doesn’t impact the rendering logic
**Best practices**
- Think of refs as an escape hatch
- Don’t read or write `ref.current` during rendering


# Manipulate DOM elements with Refs

Getting a ref to a DOM node
- Pass ref with ref attribute to a JSX tag
- When React creates DOM node for the element, React will put reference to the node into `ref.current`
Now, you can use built-in borwser APIs
- e.g. `myRef.current.scrollIntoView()`

List of refs can get complicated- one possible solution:
- `ref` callback: pass a function to the ref attribute
	- React will call ref callback with the DOM node when it sets ref- and null when it’s time to clear it
Sample code → 
```typescript
const itemsRef = useRef(null);

function getMap() {
	if (!itemsRef.current) {
	  // Initialize the Map on first usage.
	  itemsRef.current = new Map();
	}
	return itemsRef.current;
}

...

{catList.map(cat => (
	<li
	  key={cat.id}
	  ref={(node) => {
		const map = getMap();
		if (node) {
		  map.set(cat.id, node);
		} else {
		  map.delete(cat.id);
		}
	  }}
>	
	  <img
		src={cat.imageUrl}
		alt={'Cat #' + cat.id}
	  />
	</li>
))}
```
- itemsRef holds a map from item ID to DOM node 
- `ref` callback on every list item takes care to update the map

**Accessing other component’s DOM nodes**

Componens must opt in to exposing their DOM nodes
- Specify wth `forwardRef` API
```typescript
const MyInput = forwardRef((props, ref) => {
  return <input {...props} ref={ref} />;
});
```
1. `<MyInput ref={inputRef} />` tells React to put corresponding DOM node into `inputRef.current`, but  MyInput must opt in
2. `forwardRef` opts it in to recieving ref
3. Then, passes ref to input

You can limit the exposed functionality of component’s DOM using `useImperativeHandle`
For example, to only expose focus
```typescript
const MyInput = forwardRef((props, ref) => {
  const realInputRef = useRef(null);
  useImperativeHandle(ref, () => ({
    // Only expose focus and nothing else
    focus() {
      realInputRef.current.focus();
    },
  }));
  return <input {...props} ref={realInputRef} />;
});
```

**When does React attach the refs?**
Update has 2 phases- render and commit

Generall, don’t access refs during rendering (too early to rea them)
Usually- access refs from event handlers

**Flushing state updates synchronously with flushSync**
Issue: state updates are queued and you may want to use refs immedaitely
Solution: force React to update (“flush”) the DOM synhronously

How? Use `flushSync` from `react-dom` and wrap the state update into a `flushSync` call
```typescript
flushSync(() => {
  setTodos([ ...todos, newTodo]);
});
listRef.current.lastChild.scrollIntoView();
```
What does this do?
- Updates DOM write after the code in flushSync
- So- when scroll into view, the last todo will already be in the DOM

**Best practices for DOM manipulatoin with refs**
Remember: Refs are an escape hatch!
- Stick to non-destructive actions
- Avoid changing DOM nodes managed by React
- Althugh- you can safely modify parts of the DOM that React has no reason to update


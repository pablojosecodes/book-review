---
title: Escape Hatches
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

# Synchronizing with Effects
Effects’ purpose
- Run code after rendering so you can synchronize component with some system outside of React

Remember, we have (among other)
- Rendering code (must be pure)
- Event handlers- nested functions that do things. (have side effects)
However, sometimes this isn’t enough.
Example
- `ChatRoom` component- need to connect to chat server whenever it’s visible on the screen
- Issue: has side effect but not triggered by event handler
So, Effect
- Specify side effects caused by rendering itself, rather than an event

**Note: don’t rush to add Effects to your components **

**Writing an effect**

1. Declare an effect
`import { useEffect } from 'react';`
And then `useEffect(() => { /* code */ });`

Then, every time your component renders React will update screen, and then run code in `useEffect`

Issue: effects run after every render

2. Specify effect’s dependencies
Default: runs after *every* render
Add an array of dependencies to the end of `useEffect` usage
- Specifies when to run
- Must include any things you depend on within the useEffect
`[]` for example- only runs useEffect when component mounts

3. Add cleanup if needed
Return a cleanup function, whch cleans up whatever mess you might have made
e.g. →
```typescript
  useEffect(() => {
    const connection = createConnection();
    connection.connect();
    return () => connection.disconnect();
  }, []);
```


**Common patterns to deal with Effect firing twice**
Typically: try and make `setup` indistinguishable from `setup -> cleanup -> setup`

**Controlling non-React widgets**
Use case: adding UI widget whch isn’t written to React
Issue if doing something like `showModal`- don’t want to call it twice.
So →
```typescript
useEffect(() => {
  const dialog = dialogRef.current;
  dialog.showModal();
  return () => dialog.close();
}, []);
```

**Subscrbing to events**
Unsubscribe in cleanup function!

**Triggering animations**
Reset to original animation values in cleanup function

**Fetching data**
Either abort or ignore its result
Example →
```typescript
useEffect(() => {
  let ignore = false;

  async function startFetching() {
    const json = await fetchTodos(userId);
    if (!ignore) {
      setTodos(json);
    }
  }

  startFetching();

  return () => {
    ignore = true;
  };
}, [userId]);
```

**Sending Analytics**
If logging, shouldn’t need to fix

Don’t use effect for
- Initializing application
- Things like sending buying request to api
	- This is not caused by rendering
	- Caused by a specific interaction


# You might not neeed an effect
Effects are escape hatches!!
- You don’t need an effect to transform data for rending
- You don’t need effect to adjust some state based on some user event
- You do need effcts to synchronize with external systems

Some common concrete examples →

**Updating state based on props or state**
Say you have `firstName` and `lastName` and want `fullName`
Just declare `const fullName = firstName + lastName;`

**Caching expensive calculations**
Might be tempted to use useEffect for expensive things. 
Example of what NOT to do
```typescript
const [visibleTodos, setVisibleTodos] = useState([]);
useEffect(() => {
setVisibleTodos(getFilteredTodos(todos, filter));
}, [todos, filter]);
```

Instead, you can use the `useMemo` hook to cache the expensive calculation
```typescript
const visibleTodos = useMemo(() => getFilteredTodos(todos, filter), [todos, filter]);
```

**Resetting all state when a prop changes**
Say you want: reset all state when a state changes (ie switching chat boxes)
Naive implementation would be using useEffect with dependency of what’s changing
Instead-
- Tell React that each is different by passing an explicit key
- React treats them as differnt components which wouldn’t share any state

**Adjusting some state when a prop changes**
Say you want: change state when prop changes
Instead
- Adjust the state while you’re rendering using `useState`
Sample $\Downarrow$
```jsx
const [prevItems, setPrevItems] = useState(items);
  if (items !== prevItems) {
    setPrevItems(items);
    setSelection(null);
  }
```
But even more efficient to just calculate everything during rendering
```jsx
function List({ items }) {
  const [isReverse, setIsReverse] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const selection = items.find(item => item.id === selectedId) ?? null;
}
```

**Sharing logic between event handlers**
Say you want: show notification whenever user puts product in cart (2 buttons for this)
- Might be tempted to use `useEffect` for this and update based on changes triggered by either
- Unnecessary! Only use effects for code that sould run becausethe component was displayed to the user
Instead, just put shared logic into function called from both handlers


**Sending a POST request**
If related to user action and not something that was just displayed, don’t use

**Chain of Computations**
Say you want to: adjust piece of state based on another state
- DONT
- Instead, calculate what you can during rendering and adjust state in event handler

**Initializing Application**
Remember that useEffect runs twice in development- this can cause issues

**Notifying parent components about state changes**
Say you want to let a parent component know when internal state changes- so you call onchange event from effect
Instead
- Update the state of both components within the same event handler

**Passing data to parent**
Always pass data down to the child- let parent fetch the data instead

**Subscribing to external store**
Say you want to get data from outside of the React state
You can use `useSyncExternalStore` here

**Fetching data**
You can use in `useEffect` but make sure you add a cleanup function to ignore stale responses and avoid race conditions



# Lifecycle of Reactive Effects
While components
- Mount
- Update
- Unmounet
Effects only
- Start synchronizing
- Stop synchronizing

**Why synchronization may need to happen more than once**
Example: connecting to ifferent chat rooms

How react re-synchronizes effect
- Calls cleanup function
- Runs effect with new value

Example workflow from component perspective (chat room example)
1. `ChatRoom` mounts with `roomId` set to ‘general’
	1. Effect connects to “general” room
1. `ChatRoom` mounts with `roomId` set to ‘travel’
	1. Effect disconnects from “general” and connects to “travel” room
2. `ChatRoom` unmounted
	1. Effect disconnects from “travel” room

In development, React verifies that your effect can re-synchronize by forcing it to do that immediately in development

Use separate effects for separate and independent synchronization processes

Remember, **reactive** values must be included in depndencies
- Props, state, other values in component
- Note: all varibables in the comonent body are reactive
	- Can move out of component or into useEffect if don’t want as dependencies
- Mutable values aren’t reactive


Pay attention to linter! Always a way to fix the code


# Separating events from Effects
(BETA- more details here that I didn’t cover)
Sometimes you want a mix of behavrio that’s event handler like but also like effect
- Effect which re-runs in response to some values but not other
- `useEffectEvents`lets you do that


# Removing Effect Dependencies

Linter verifies you have all reactive vlues that the effect reads, in the list of dependencies
BUT- don’t want unnecessary dependencies

**Core idea**: prove a dependency is not a dependency in order to remove it.
- You don’t “choose” your dependencies- each reactive value used must be in there

Check out dependencies- does it make sense for effect to re-run when any of the dependencies change?

Things to think about $\Downarrow$ 

**Should this code be an effect at all?**
If responding to a particular interaction (ie. click)
- Put logic into event handler instead

**Is effect doing unrelated things?**
Make sure code is atomized, with logic running when needed for its dependencies

**Are you reading some state to calculate the next state?**
If modifying state but don’t want state itself to be a dependency, use an updater function instead

**Want to read value without reacting to its changes?**
(BETA- didn’t add info here)







# Reusing logic with Custom Hooks

Use cusotm hooks if you wants specific hook
Examples of possible use cases
- Fetch data
- Keep track of whether use is online
- Connect to chat room
- Pointer position


Imagine you have a useEffect with state, that you keep reusing
- Example- check if wifi connection is enabled

Well, you can reuse.
Steps
1. Define the functionality (using  possibly `useState` and `useEffect`)
2. Put the code into its own function and return the state

Sample code $\Downarrow$ 
```typescript
function useOnlineStatus() {
  const [isOnline, setIsOnline] = useState(true);
  useEffect(() => {
    function handleOnline() {
      setIsOnline(true);
    }
    function handleOffline() {
      setIsOnline(false);
    }
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);
  return isOnline;
}
```

Rules
1. Names start with “use” followd by capital letter
2. Need to be pure

Notes: custom hooks don’t share state, just stateful logic (ie. wouldn’t have the same `isOnline` value by default across various usages)
- Completely independent instances when used


**Passing Event Handlers to Custom Hooks**

Custom hooks can accept event handlers
Wrap the event handlers in `useEffectEvent` to remove from the dependencies


**When to use Custom Hooks**
With time, put most of apps effects in custom hooks
Up to you where to draw the boundaries

Replace with the hooks from React when possible (consider more edge cases and whatnot)

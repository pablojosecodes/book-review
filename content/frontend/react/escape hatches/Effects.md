---
title: Effects
---
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








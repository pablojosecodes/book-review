---
title: State Update Operations
---


# Queueing a Series of State Updates
Sometims ou want to perform multiple operations in state before queueing the next render


React batches state updates
- Can’t just call `setState(state+1)` 10 times in a row (would be the same as once)
- **Batching** React waits until all code in event handlers has run before processing your state updates (viz. waiter taking your order)

Updating same state multiple times
- If you needed to update variable multiple times before newxt render, you can pass an `updater` function instead
	- `setNumber(n => n + 1)`
- React still goes through the state queue, but now it doesn’t rely on old state

# Updating Objects in State

You can store any Javscript value in state including objects
- Treat objects as immutable- numbers numbres, booleans, strings
Basically- **treat state as read-only** 

Spread syntax
- ie. `setPerson({...person, firstname: e.target.value});`
- Also, you can use braces inside object definition to specify property with dynamic name
	- ie. `setPerson({...person, [e.target.name]: e.target.value});`
- Note: spread syntax is shallow

Updating a nested object
- ie. `setPerson({...person, artwork: {...person.artwork, city: 'New Delhi'}});`
- Side note: objets aren’t actualy nested 
	- “nested objects” are objects which point to other objects
- Shortcut- **Immer**
	- Lets nested update look more like `updatePerson(draft => {draft.artwork.city = 'Lagos';})`

# Updating Arrays in State
Treat arrays (like objects) as read-only!
- Need to pass a new array in state setting function
- Note: slice vs splice
	- `splice`- mutate array
	- `slice` - copy array or part of it

Add to array (spread)
- `setArtists([...artists, {id: nextId++, name: name}]);`
Removing from array (filter)
- `setArtists(artists.filter(a => a.id !== artist.id));`
Changing all or some elements (map)
```typescript

const nextShapes = shapes.map(shape => {
  if (shape.type === 'square') {
	// No change
	return shape;
  } else {
	// Return a new circle 50px below
	return {
	  ...shape,
	  y: shape.y + 50,
	};
  }
});
setShapes(nextShapes);
```

Replacing items in array (map)
Inserting into an array (spread and slice())
```typescript
    const nextArtists = [
      // Items before the insertion point:
      ...artists.slice(0, insertAt),
      // New item:
      { id: nextId++, name: name },
      // Items after the insertion point:
      ...artists.slice(insertAt)
    ];
    setArtists(nextArtists);
```

Other more complicated changes
- Can always just `const newArray = [...oldarray];`and then make the changes
Updating objects inside arrays
- Need to create copies from teh point where you want to update, and all the way up to the top lvel

In general, ony mutate objects you have just created

Using Immer:
- Makes it easy to update nested arrays without mutation


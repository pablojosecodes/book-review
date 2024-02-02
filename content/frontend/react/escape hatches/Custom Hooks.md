---
title: Custom Hooks
---
# Reusing logic with Custom Hooks

Use custom hooks if you wants specific hook
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

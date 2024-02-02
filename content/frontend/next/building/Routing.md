---
title: Routing
---
# Basics
Skeleton of every application

Some terms
- Tree-=
- Subtree- part of tree starting at a new root and ending at leaves
- Root- first node in tree/substree
- Leaf- nodes in subtree with no children
- URL Segment- part of url delimited by slashes
- URL Path- url past domain


**Folders**
- Folders: define routes
	- You can nest them
- File: create the UI

Colocation: you can also put components, styues, etc. in folders in app directory
- Only content from `page` or `route` is publicaly addressable



# routes + pages/layouts

**Creating Routes**
- Folder = route segment
- `page.tsx` makes it publically accessible

****
**Page**: unique to a route
- Makes route publically accessible
- Server Components by default
- Can fetch data

**Layout**: UI shared between multiple pages
- On navigation, preserve state, remain interactive, and don’t re-render
- Server Components by default
- Can fetch data
- Can nest them

**Templates**
- Wrap each child layout or page
- Create new instance for each of their children on navigation
- When to use instead of layout?
	- Features which rely on `useEffect` and `useState` on a per-page basis
- Defining- use `template` file and accept a `children` prop
- Rendered between Layout and children


# Linking / Navigation

3 ways to navigate between routes in NextJS
- `<Link>`
- `useRouter`
- History API

`<Link>`: built in component which extends the HTML `<a>` tag to provide prefetching and client-side navigation between routes
- Primary/recommended navigation way
- Use
	- Import from `next/link`
	- Pass `href` prop
- Specific cases
	- Scroll to specific ID- `#id` in `href`
	- Check if is current link- `usePathName()` from `next/navigation`
	- Disable scroll behavior- pass `scroll-{false}` to `<Link>`

`useRouter()`: programmatically change routes from Client Components (use `redirect()` in server components instead)
- Basic usage: `onClick={() => router.push('/path')}`

**Native History API**
- Can use the native `window.history.pushState` and `window.history.replaceState` methods to update history stack without reloading the page
- `window.history.pushState`: add new entyr to browser’s history stack (e.g. product adding)
- `window.history.replaceState` to replace the current entry on the browser’s ihistory stack (user can’t navigate to previous state)
	- Example use case: switch locale

## How Routing and Navigation Works
App router uses hybrid approach for routing and navigation
- Server side: application code auomatically code-split by route segments
- Client side: Prefetch and cache route segments


1. Code Splitting: lets you split app code into smaller bundles to be downloaded/executed by browser
2. Prefetching: preload route in background before user visits it. 2 methods
	1. `<Link>` component
		1. Automatically prefetched
	2. `router.prefetch()`
3. Caching: Router cache
	1. As users navigate around app, prefetched route segments / visited orutes are stored in the cache
4. Partial rendering: only route segments which change on navigation re-render on the client
	1. For example, top level layout doesn’t re-render
5. Back and forward navigation: by default, nextjs
	1. Maintains scroll position
	2. Re-uses route segments in router cache


# Loading UI and Streaming

TODO

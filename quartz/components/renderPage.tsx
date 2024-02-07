import { render } from "preact-render-to-string"
import { QuartzComponent, QuartzComponentProps } from "./types"
import HeaderConstructor from "./Header"
import BodyConstructor from "./Body"
import { JSResourceToScriptElement, StaticResources } from "../util/resources"
import { FullSlug, RelativeURL, joinSegments, normalizeHastElement } from "../util/path"
import { visit } from "unist-util-visit"
import { Root, Element, ElementContent } from "hast"
import { QuartzPluginData } from "../plugins/vfile"

interface RenderComponents {
  head: QuartzComponent
  header: QuartzComponent[]
  beforeBody: QuartzComponent[]
  pageBody: QuartzComponent
  left: QuartzComponent[]
  right: QuartzComponent[]
  footer: QuartzComponent
}

export function pageResources(
  baseDir: FullSlug | RelativeURL,
  staticResources: StaticResources,
): StaticResources {
  const contentIndexPath = joinSegments(baseDir, "static/contentIndex.json")
  const contentIndexScript = `const fetchData = fetch("${contentIndexPath}").then(data => data.json())`

  return {
    css: [joinSegments(baseDir, "index.css"), ...staticResources.css],
    js: [
      {
        src: joinSegments(baseDir, "prescript.js"),
        loadTime: "beforeDOMReady",
        contentType: "external",
      },
      {
        loadTime: "beforeDOMReady",
        contentType: "inline",
        spaPreserve: true,
        script: contentIndexScript,
      },
      ...staticResources.js,
      {
        src: joinSegments(baseDir, "postscript.js"),
        loadTime: "afterDOMReady",
        moduleType: "module",
        contentType: "external",
      },
    ],
  }
}

let pageIndex: Map<FullSlug, QuartzPluginData> | undefined = undefined
function getOrComputeFileIndex(allFiles: QuartzPluginData[]): Map<FullSlug, QuartzPluginData> {
  if (!pageIndex) {
    pageIndex = new Map()
    for (const file of allFiles) {
      pageIndex.set(file.slug!, file)
    }
  }

  return pageIndex
}

const themingScript = `
document.addEventListener('DOMContentLoaded', function() {
  function applyLanguageTheme() {
    const path = window.location.pathname;
    console.log(path)
    if (path.includes('languages')) {
      console.log("INCLUDES")
      // Define the colors for light and dark modes
      const lightModeColors = {
        light: "hsl(0, 0%, 100%)",       // Assuming --background corresponds to 'light'
        lightgray: "hsl(20, 14.3%, 4.1%)", // Assuming --foreground corresponds to 'lightgray'
        gray: "hsl(0, 0%, 100%)",         // Assuming --card corresponds to 'gray'
        darkgray: "hsl(20, 14.3%, 4.1%)", // Assuming --card-foreground corresponds to 'darkgray'
        dark: "hsl(60, 4.8%, 95.9%)",     // Assuming --secondary corresponds to 'dark'
        secondary: "hsl(47.9, 95.8%, 53.1%)", // --primary
        tertiary: "hsl(26, 83.3%, 14.1%)",    // --primary-foreground
        highlight: "hsl(60, 4.8%, 95.9%)"   // Assuming --accent corresponds to 'highlight'
    };

      const darkModeColors = {
        light: "hsl(20, 14.3%, 4.1%)",     // Assuming --background corresponds to 'light'
        // lightgray: "hsl(60, 9.1%, 97.8%)", // Assuming --foreground corresponds to 'lightgray'
        lightgray: "#404040",
        gray: "hsl(20, 14.3%, 4.1%)",       // Assuming --card corresponds to 'gray'
        darkgray: "hsl(60, 9.1%, 97.8%)",   // Assuming --card-foreground corresponds to 'darkgray'
        // dark: "hsl(12, 6.5%, 15.1%)",       // Assuming --secondary corresponds to 'dark'
        dark:"white",
        secondary: "hsl(47.9, 95.8%, 53.1%)", // --primary
        tertiary: "hsl(26, 83.3%, 14.1%)",    // --primary-foreground
        highlight: "hsl(12, 6.5%, 15.1%)"   // Assuming --accent corresponds to 'highlight'
    };
    
      // Function to apply colors
      function setColors(colors) {
        Object.keys(colors).forEach(key => {
          document.documentElement.style.setProperty('--' + key, colors[key]);
        });
      }

      // Check if dark mode is active and apply the corresponding colors
      const savedTheme = document.documentElement.getAttribute('saved-theme') || 'light';
      if (savedTheme === 'dark') {
        setColors(darkModeColors);
        console.log("SETTING DARK MODE COLORS")
        console.log(darkModeColors)
      } else {
        console.log("SETTING LIGHT MODE COLORS")
        setColors(lightModeColors);
      }
    }
  }
  applyLanguageTheme();
});
`;


export function renderPage(
  slug: FullSlug,
  componentData: QuartzComponentProps,
  components: RenderComponents,
  pageResources: StaticResources,
): string {
  // console.log("HI")
  // process transcludes in componentData
  visit(componentData.tree as Root, "element", (node, _index, _parent) => {
    if (node.tagName === "blockquote") {
      const classNames = (node.properties?.className ?? []) as string[]
      if (classNames.includes("transclude")) {
        const inner = node.children[0] as Element
        const transcludeTarget = inner.properties["data-slug"] as FullSlug
        const page = getOrComputeFileIndex(componentData.allFiles).get(transcludeTarget)
        if (!page) {
          return
        }

        let blockRef = node.properties.dataBlock as string | undefined
        if (blockRef?.startsWith("#^")) {
          // block transclude
          blockRef = blockRef.slice("#^".length)
          let blockNode = page.blocks?.[blockRef]
          if (blockNode) {
            if (blockNode.tagName === "li") {
              blockNode = {
                type: "element",
                tagName: "ul",
                properties: {},
                children: [blockNode],
              }
            }

            node.children = [

              normalizeHastElement(blockNode, slug, transcludeTarget),
              {
                type: "element",
                tagName: "a",
                properties: { href: inner.properties?.href, class: ["internal"] },
                children: [{ type: "text", value: `Link to original` }],
              },
            ]
          }
        } else if (blockRef?.startsWith("#") && page.htmlAst) {
          // header transclude
          blockRef = blockRef.slice(1)
          let startIdx = undefined
          let endIdx = undefined
          for (const [i, el] of page.htmlAst.children.entries()) {
            if (el.type === "element" && el.tagName.match(/h[1-6]/)) {
              if (endIdx) {
                break
              }

              if (startIdx !== undefined) {
                endIdx = i
              } else if (el.properties?.id === blockRef) {
                startIdx = i
              }
            }
          }

          if (startIdx === undefined) {
            return
          }

          node.children = [
            ...(page.htmlAst.children.slice(startIdx, endIdx) as ElementContent[]).map((child) =>
              normalizeHastElement(child as Element, slug, transcludeTarget),
            ),
            {
              type: "element",
              tagName: "a",
              properties: { href: inner.properties?.href, class: ["internal"] },
              children: [{ type: "text", value: `Link to original` }],
            },
          ]
        } else if (page.htmlAst) {
          // page transclude
          node.children = [
            {
              type: "element",
              tagName: "h1",
              properties: {},
              children: [
                { type: "text", value: page.frontmatter?.title ?? `Transclude of ${page.slug}` },
              ],
            },
            ...(page.htmlAst.children as ElementContent[]).map((child) =>
              normalizeHastElement(child as Element, slug, transcludeTarget),
            ),
            {
              type: "element",
              tagName: "a",
              properties: { href: inner.properties?.href, class: ["internal"] },
              children: [{ type: "text", value: `Link to original` }],
            },
          ]
        }
      }
    }
  })

  const {
    head: Head,
    header,
    beforeBody,
    pageBody: Content,
    left,
    right,
    footer: Footer,
  } = components
  const Header = HeaderConstructor()
  const Body = BodyConstructor()

  const LeftComponent = (
    <div class="left sidebar">
      {left.map((BodyComponent) => (
        <BodyComponent {...componentData} />
      ))}
    </div>
  )

  const RightComponent = (
    <div class="right sidebar">
      {right.map((BodyComponent) => (
        <BodyComponent {...componentData} />
      ))}
    </div>
  )



  const themingScriptElement = {
    type: "element",
    tagName: "script",
    properties: {},
    children: [{ type: "text", value: themingScript }],
  };



  const doc = (
    <html>
      <Head {...componentData} />
      <body data-slug={slug}>
        <div id="quartz-root" class="page">
          <Body {...componentData}>
            {LeftComponent}
            <div class="center">
              <div class="page-header">
                <Header {...componentData}>
                  {header.map((HeaderComponent) => (
                    <HeaderComponent {...componentData} />
                  ))}
                </Header>
                <div class="popover-hint">
                  {beforeBody.map((BodyComponent) => (
                    <BodyComponent {...componentData} />
                  ))}
                </div>
              </div>
              <Content {...componentData} />
            </div>
            {RightComponent}

          </Body>
          <Footer {...componentData} />
        </div>
        <script dangerouslySetInnerHTML={{ __html: themingScript }}></script>

      </body>

      {pageResources.js
        .filter((resource) => resource.loadTime === "afterDOMReady")
        .map((res) => JSResourceToScriptElement(res))}
    </html>
  )

  return "<!DOCTYPE html>\n" + render(doc)
}

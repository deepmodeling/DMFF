# 4. Document writing

The most important thing after you implement your code in DMFF is to write the docs. Good documentation can help users use your module correctly and allow other maintainers to improve functions better.

The documentation of DMFF use [MKDocs](https://www.mkdocs.org/) framework. 

## 4.1 Install MKDocs

Before we start to write our docs, run the following command in termianl to install MKDocs:

```
pip install mkdocs
```

To support latex rendering, we also need markdown extension:

```
pip install pymdown-extensions
```

## 4.2 Write your docs

According to the existing document architecture, create a new markdown file in the appropriate directory. Write it! If you need to insert picture, upload the picture in `assets` directory, and use `![_placeholder](relative/path/to/this/file)` syntax to insert picture.

## 4.3 Preview you docs

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it. Make sure you're in the same directory as the `mkdocs.yml` configuration file, and then start the server by running the mkdocs serve command:

```
mkdocs serve
```

Open up http://127.0.0.1:8000/ in your browser, and you'll see the default home page being displayed. The dev-server also supports auto-reloading, and will rebuild your documentation whenever anything in the configuration file, documentation directory, or theme directory changes.
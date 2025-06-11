# fmt: off

"""WSGI Flask-app for browsing a database.

::

    +---------------------+
    | layout.html         |
    | +-----------------+ |    +--------------+
    | | search.html     | |    | layout.html  |
    | |     +           | |    | +---------+  |
    | | table.html ----------->| |row.html |  |
    | |                 | |    | +---------+  |
    | +-----------------+ |    +--------------+
    +---------------------+

You can launch Flask's local webserver like this::

    $ ase db abc.db -w

or this::

    $ python3 -m ase.db.app abc.db

"""

import io
import sys
from pathlib import Path

from ase.db import connect
from ase.db.core import Database
from ase.db.project import DatabaseProject
from ase.db.web import Session


class DBApp:
    root = Path(__file__).parent.parent.parent

    def __init__(self):
        self.projects = {}

        flask = new_app(self.projects)
        self.flask = flask

        # Other projects will want to control the routing of the front
        # page, for which reasons we route it here in DBApp instead of
        # already in new_app().
        @flask.route('/')
        def frontpage():
            projectname = next(iter(self.projects))
            return flask.view_functions['search'](projectname)

    def add_project(self, name: str, db: Database) -> None:
        self.projects[name] = DatabaseProject.load_db_as_ase_project(
            name=name, database=db)

    @classmethod
    def run_db(cls, db):
        app = cls()
        app.add_project('default', db)
        app.flask.run(host='0.0.0.0', debug=True)


def new_app(projects):
    from flask import Flask, render_template, request
    app = Flask(__name__, template_folder=str(DBApp.root))

    @app.route('/<project_name>')
    @app.route('/<project_name>/')
    def search(project_name: str):
        """Search page.

        Contains input form for database query and a table result rows.
        """
        if project_name == 'favicon.ico':
            return '', 204, []  # 204: "No content"
        session = Session(project_name)
        project = projects[project_name]
        return render_template(str(project.get_search_template()),
                               q=request.args.get('query', ''),
                               project=project,
                               session_id=session.id)

    @app.route('/update/<int:sid>/<what>/<x>/')
    def update(sid: int, what: str, x: str):
        """Update table of rows inside search page.

        ``what`` must be one of:

        * query: execute query in request.args (x not used)
        * limit: set number of rows to show to x
        * toggle: toggle column x
        * sort: sort after column x
        * page: show page x
        """
        session = Session.get(sid)
        project = projects[session.project_name]
        session.update(what, x, request.args, project)
        table = session.create_table(project.database,
                                     project.uid_key,
                                     keys=list(project.key_descriptions))
        return render_template(str(project.get_table_template()),
                               table=table,
                               project=project,
                               session=session)

    @app.route('/<project_name>/row/<uid>')
    def row(project_name: str, uid: str):
        """Show details for one database row."""
        project = projects[project_name]
        row = project.uid_to_row(uid)
        dct = project.row_to_dict(row)
        return render_template(str(project.get_row_template()),
                               dct=dct, row=row, project=project, uid=uid)

    @app.route('/atoms/<project_name>/<int:id>/<type>')
    def atoms(project_name: str, id: int, type: str):
        """Return atomic structure as cif, xyz or json."""
        row = projects[project_name].database.get(id=id)
        a = row.toatoms()
        if type == 'cif':
            b = io.BytesIO()
            a.pbc = True
            a.write(b, 'cif', wrap=False)
            return b.getvalue(), 200, []

        fd = io.StringIO()
        if type == 'xyz':
            a.write(fd, format='extxyz')
        elif type == 'json':
            con = connect(fd, type='json')
            con.write(row,
                      data=row.get('data', {}),
                      **row.get('key_value_pairs', {}))
        else:
            1 / 0

        headers = [('Content-Disposition',
                    'attachment; filename="{project_name}-{id}.{type}"'
                    .format(project_name=project_name, id=id, type=type))]
        txt = fd.getvalue()
        return txt, 200, headers

    @app.route('/gui/<int:id>')
    def gui(id: int):
        """Pop ud ase gui window."""
        from ase.visualize import view

        # XXX so broken
        arbitrary_project = next(iter(projects))
        atoms = projects[arbitrary_project].database.get_atoms(id)
        view(atoms)
        return '', 204, []

    @app.route('/test')
    def test():
        return 'hello, world!'

    @app.route('/robots.txt')
    def robots():
        return ('User-agent: *\n'
                'Disallow: /\n'
                '\n'
                'User-agent: Baiduspider\n'
                'Disallow: /\n'
                '\n'
                'User-agent: SiteCheck-sitecrawl by Siteimprove.com\n'
                'Disallow: /\n',
                200)

    return app


def main():
    db = connect(sys.argv[1])
    DBApp.run_db(db)


if __name__ == '__main__':
    main()
